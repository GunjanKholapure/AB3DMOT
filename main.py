from __future__ import print_function
import os.path, copy, numpy as np, time, sys
from numba import jit
from sklearn.utils.linear_assignment_ import linear_assignment
from filterpy.kalman import KalmanFilter
from utils import load_list_from_folder, fileparts, mkdir_if_missing
from scipy.spatial import ConvexHull
import uuid
import argoverse
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
import json
import argparse

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--dthresh', type=float, default=0.2, help='detection threshold')

parser.add_argument('--mal', type=int, default=3, help='max age lower value for termination of tracks')
parser.add_argument('--mah', type=int, default=3, help='max age higher value for termination of tracks')

parser.add_argument('--min_hits', type=int, default=3, help='min hits for initializing new tracks')
parser.add_argument('--ithresh', type=float, default=0.1, help='iou threshold for mathing tracks to detections')
parser.add_argument('--cat', type=str, default="VEHICLE", help='category vehicle/pedestrian')
parser.add_argument('--set',type=str,default="val",help='val/train/test')
parser.add_argument('--keep_age',type=int,default=3,help='age for whoch to keep the track')
parser.add_argument('--svel', type=float, default=0.3, help='iou threshold for mathing tracks to detections')
parser.add_argument('--sdis', type=float, default=2, help='iou threshold for mathing tracks to detections')



#parser.add_argument('--res_path', type=str, default="./argo/data/car_3d_det_test/", help='result path to find tracking results on.')

args,unknown = parser.parse_known_args()




@jit    
def poly_area(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

@jit        
def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

@jit       
def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  
    
    
    
def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.
   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**

   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

def iou3d(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])
    inter_vol = inter_area * max(0.0, ymax-ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d

@jit       
def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

def convert_3dbox_to_8corner(bbox3d_input):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis
    bbox3d = copy.copy(bbox3d_input)

    R = roty(bbox3d[3])    

    # 3d bounding box dimensions
    l = bbox3d[4]
    w = bbox3d[5]
    h = bbox3d[6]
    
    # 3d bounding box corners
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [0,0,0,0,-h,-h,-h,-h];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + bbox3d[0]
    corners_3d[1,:] = corners_3d[1,:] + bbox3d[1]
    corners_3d[2,:] = corners_3d[2,:] + bbox3d[2]
 
    return np.transpose(corners_3d)

def generate_ID():
    """
    Generate tracking IDs to identify objects
    """
    return str(uuid.uuid1())

class KalmanBoxTracker(object):
  """
  This class represents the internel state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self, bbox3D, info,ids=None):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=10, dim_z=7)       
    self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix
                          [0,1,0,0,0,0,0,0,1,0],
                          [0,0,1,0,0,0,0,0,0,1],
                          [0,0,0,1,0,0,0,0,0,0],  
                          [0,0,0,0,1,0,0,0,0,0],
                          [0,0,0,0,0,1,0,0,0,0],
                          [0,0,0,0,0,0,1,0,0,0],
                          [0,0,0,0,0,0,0,1,0,0],
                          [0,0,0,0,0,0,0,0,1,0],
                          [0,0,0,0,0,0,0,0,0,1]])     
    
    self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      # measurement function,
                          [0,1,0,0,0,0,0,0,0,0],
                          [0,0,1,0,0,0,0,0,0,0],
                          [0,0,0,1,0,0,0,0,0,0],
                          [0,0,0,0,1,0,0,0,0,0],
                          [0,0,0,0,0,1,0,0,0,0],
                          [0,0,0,0,0,0,1,0,0,0]])

    # with angular velocity
    # self.kf = KalmanFilter(dim_x=11, dim_z=7)       
    # self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0,0],      # state transition matrix
    #                       [0,1,0,0,0,0,0,0,1,0,0],
    #                       [0,0,1,0,0,0,0,0,0,1,0],
    #                       [0,0,0,1,0,0,0,0,0,0,1],  
    #                       [0,0,0,0,1,0,0,0,0,0,0],
    #                       [0,0,0,0,0,1,0,0,0,0,0],
    #                       [0,0,0,0,0,0,1,0,0,0,0],
    #                       [0,0,0,0,0,0,0,1,0,0,0],
    #                       [0,0,0,0,0,0,0,0,1,0,0],
    #                       [0,0,0,0,0,0,0,0,0,1,0],
    #                       [0,0,0,0,0,0,0,0,0,0,1]])     
    
    # self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0,0],      # measurement function,
    #                       [0,1,0,0,0,0,0,0,0,0,0],
    #                       [0,0,1,0,0,0,0,0,0,0,0],
    #                       [0,0,0,1,0,0,0,0,0,0,0],
    #                       [0,0,0,0,1,0,0,0,0,0,0],
    #                       [0,0,0,0,0,1,0,0,0,0,0],
    #                       [0,0,0,0,0,0,1,0,0,0,0]])

    # self.kf.R[0:,0:] *= 10.   # measurement uncertainty
    self.kf.P[7:,7:] *= 1000. #state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
    self.kf.P *= 10.
   
    
    #self.kf.Q[-1,-1] *= 0.01    # process uncertainty
    self.kf.Q[7:,7:] *= 0.01
    self.kf.x[:7] = bbox3D.reshape((7, 1))
    
    
    self.time_since_update = 0
    
    if ids is not None:
        self.id = ids[0]
        self.unique_id = ids[1]
    else:
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.unique_id = generate_ID()
    self.history = []
    self.hits = 1           # number of total hits including the first detection
    self.hit_streak = 1     # number of continuing hit considering the first detection
    self.first_continuing_hit = 1
    self.still_first = True
    self.age = 0
    self.info = info        # other info
    self.tracked = True

  def update(self, bbox3D, info): 
    """ 
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1          # number of continuing hit
    if self.still_first:
      self.first_continuing_hit += 1      # number of continuing hit in the fist time
    
    ######################### orientation correction
    if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
    if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

    new_theta = bbox3D[3]
    if new_theta >= np.pi: new_theta -= np.pi * 2    # make the theta still in the range
    if new_theta < -np.pi: new_theta += np.pi * 2
    bbox3D[3] = new_theta

    predicted_theta = self.kf.x[3]
    if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:     # if the angle of two theta is not acute angle
      self.kf.x[3] += np.pi       
      if self.kf.x[3] > np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
      if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
      
    # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
    if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
      if new_theta > 0: self.kf.x[3] += np.pi * 2
      else: self.kf.x[3] -= np.pi * 2
    
    ######################### 

    self.kf.update(bbox3D)

    if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
    if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
    self.info = info

  def predict(self):       
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    self.kf.predict()      
    if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
    if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
      self.still_first = False
    self.time_since_update += 1
    self.history.append(self.kf.x)
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return self.kf.x
    #return self.kf.x[:7].reshape((7, ))
        

def associate_detections_to_trackers(detections,trackers,iou_threshold=args.ithresh):
# def associate_detections_to_trackers(detections,trackers,iou_threshold=0.01):     # ablation study
# def associate_detections_to_trackers(detections,trackers,iou_threshold=0.25):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  detections:  N x 8 x 3
  trackers:    M x 8 x 3

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,8,3),dtype=int)    
  iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

  for d,det in enumerate(detections):
    for t,trk in enumerate(trackers):
      iou_matrix[d,t] = iou3d(det,trk)[0]             # det: 8 x 3, trk: 8 x 3
  matched_indices = linear_assignment(-iou_matrix)      # hougarian algorithm

  unmatched_detections = []
  for d,det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t,trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0],m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class AB3DMOT(object):
  def __init__(self,min_hits=args.min_hits):      # max age will preserve the bbox does not appear no more than 2 frames, interpolate the detection
  # def __init__(self,max_age=3,min_hits=3):        # ablation study
  # def __init__(self,max_age=1,min_hits=3):      
  # def __init__(self,max_age=2,min_hits=1):      
  # def __init__(self,max_age=2,min_hits=5):      
    """              
    """
    self.max_age_far = args.mal
    self.max_age_near = args.mah
    self.max_age = self.max_age_near
    self.keep_age = args.keep_age
    self.min_hits = min_hits
    print("max_age", self.max_age,"min_hits",self.min_hits)
    self.trackers = []
    self.frame_count = 0
    self.reorder = [3, 4, 5, 6, 2, 1, 0]
    self.reorder_back = [6, 5, 4, 0, 1, 2, 3]
    self.static_locs, self.static_ids = [],[]
    self.track_status = {}

  def update(self,dets_all,center_kittif):
    """
    Params:
      dets_all: dict
        dets - a numpy array of detections in the format [[x,y,z,theta,l,w,h],[x,y,z,theta,l,w,h],...]
        info: a array of other info for each det
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    tracked = []
    dets, info = dets_all['dets'], dets_all['info']         # dets: N x 7, float numpy array
    dets = dets[:, self.reorder]
    self.frame_count += 1

    trks = np.zeros((len(self.trackers),7))         # N x 7 , #get predicted locations from existing trackers.
    to_del = []
    ret = []
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict().reshape((-1, 1))
      trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]       
      if(np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))   
    for t in reversed(to_del):
      self.trackers.pop(t)

    dets_8corner = [convert_3dbox_to_8corner(det_tmp) for det_tmp in dets]
    if len(dets_8corner) > 0: dets_8corner = np.stack(dets_8corner, axis=0)
    else: dets_8corner = []
    trks_8corner = [convert_3dbox_to_8corner(trk_tmp) for trk_tmp in trks]
    
    
    if len(trks_8corner) > 0: 
      trks_8corner = np.stack(trks_8corner, axis=0)
    
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner, trks_8corner, iou_threshold=args.ithresh)
  
    
    #update matched trackers with assigned detections
    for t,trk in enumerate(self.trackers):
      if t not in unmatched_trks:
        d = matched[np.where(matched[:,1]==t)[0],0]     # a list of index
        trk.update(dets[d,:][0], info[d, :][0])
        trk.tracked = True
      else:
        trk.tracked = False

    #create and initialise new trackers for unmatched detections
    for i in unmatched_dets:        # a scalar of index
        mv_loc = [dets[i,0],dets[i,2]]
        marg,mval = 0,1000
        
        for snum,val in enumerate(self.static_locs):
            #print(mv_loc,val)
            check_dist = np.linalg.norm(np.array(mv_loc - val ))
            if check_dist < mval:
                marg = snum
                mval = check_dist
        #print(self.static_ids)
        if marg < len(self.static_ids):
            frame_diff = self.frame_count - self.static_ids[marg][2]
        
        # 
        if marg < len(self.static_ids) and (mval <  args.sdis )  and self.track_status[self.static_ids[marg][0]]==0:
            #print(mv_loc,self.static_locs[marg])
            trk = KalmanBoxTracker(dets[i,:], info[i, :],ids=self.static_ids[marg])
            self.track_status[self.static_ids[marg][0]] = 1
        else:
            trk = KalmanBoxTracker(dets[i,:], info[i,:])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        tracker_state = trk.kf.x
        d = tracker_state[:7].reshape((7,))      # bbox location
        vel_arr = [tracker_state[7],tracker_state[9]]
        d = d[self.reorder_back]
        curpos = np.array([d[3],d[5]])
        velocity = np.linalg.norm(np.array([vel_arr]))
        #if velocity <0.1:
        #    print(velocity,d[3]//1,d[5]//1,trk.id)
        dist = np.linalg.norm(curpos-center_kittif)
        #print(dist)
        
        if dist > 50:
            self.max_age = self.max_age_far
        else:
            self.max_age = self.max_age_near
        
        if((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):      
          ret.append(  ( np.concatenate((d, [trk.id+1],trk.info)), trk.unique_id, trk.tracked  ) ) # +1 as MOT benchmark requires positive
        i -= 1
        #remove dead tracklet
        if(trk.time_since_update >= self.keep_age):
          #print(velocity)
          if velocity < args.svel:
            self.static_locs.append( np.array([d[3],d[5]]) )
            self.static_ids.append( (trk.id,trk.unique_id,self.frame_count,velocity) )
            self.track_status[trk.id] = 0
          self.trackers.pop(i)
    if(len(ret)>0):
      return ret     # x, y, z, theta, l, w, h, ID, other info, confidence
    return np.empty((0,16))      

def euler_to_quaternion(r):
    (yaw, pitch, roll) = (r[0], r[1], r[2])
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

## change threshhold score, paths, category.    
if __name__ == '__main__':
  '''
  if len(sys.argv)!=2:
    print("Usage: python main.py result_sha(e.g., car_3d_det_test)")
    sys.exit(1)
  '''
  result_sha = sys.argv[1]
  save_root = './results'

  det_id2str = {1:'Pedestrian', 2:'Car', 3:'Cyclist'}
  seq_file_list, num_seq = load_list_from_folder(os.path.join('data/argo', result_sha))
  total_time = 0.0
  total_frames = 0
  save_dir = os.path.join(save_root, result_sha); mkdir_if_missing(save_dir)
  eval_dir = os.path.join(save_dir, 'data'); mkdir_if_missing(eval_dir)
  root_dir = "./../argodataset/argoverse-tracking/" + args.set
  argoverse_loader = ArgoverseTrackingLoader(root_dir)
  sname = result_sha[result_sha.find("det_") + 4:]
  argo_save_name = sname + "_t" + str(args.dthresh*100) + "_a" + str(args.mal) + str(args.mah) + "_h" + str(args.min_hits) + "_ioudot" + str(args.ithresh*100) + "_ka" + str(args.keep_age) + "_v" + str(args.svel) + "_d" + str(args.sdis)  

  print(argo_save_name)
  for seq_file in seq_file_list:
    _, seq_name, _ = fileparts(seq_file)
    #if seq_name != "6db21fda-80cd-3f85-b4a7-0aadeb14724d":
    #    continue
    print(seq_name)
    argoverse_data = argoverse_loader.get(seq_name)
    city_name = argoverse_data.city_name
    
    mot_tracker = AB3DMOT() 
    seq_dets = np.loadtxt(seq_file, delimiter=',') #load detections
    eval_file = os.path.join(eval_dir, seq_name + '.txt'); eval_file = open(eval_file, 'w')
    save_trk_dir = os.path.join(save_dir, 'trk_withid', seq_name); mkdir_if_missing(save_trk_dir)
    print("Processing %s." % (seq_name))
    track_test_res_path = "./argo_results/" + argo_save_name
    
    if not os.path.exists(track_test_res_path):
        os.mkdir(track_test_res_path)
        
    track_test_res_path1 =  os.path.join(track_test_res_path ,seq_name)
    track_test_res_path2 = os.path.join(track_test_res_path1,"per_sweep_annotations_amodal/")
    
    if not os.path.exists(track_test_res_path1):
        os.mkdir(track_test_res_path1)
        if not os.path.exists(track_test_res_path2):
            os.mkdir(track_test_res_path2)
        else:
            for files in os.listdir(track_test_res_path2):
                os.remove(track_test_res_path2+files)
                                
    
    print("filter_Score",args.dthresh)
                                  
    for frame in range(int(seq_dets[:,0].min()), int(seq_dets[:,0].max()) + 1):
      #print(frame)
      city_to_egovehicle_se3 = argoverse_data.get_pose(frame)
      center_coords = city_to_egovehicle_se3.transform_point_cloud(np.array([[0,0,0]]))[0]
      center_kittif = np.array([-center_coords[1], center_coords[0] ] )
      track_list = []
      lidar_timestamp = argoverse_data.lidar_timestamp_list[frame]
      save_trk_file = os.path.join(save_trk_dir, '%06d.txt' % frame); save_trk_file = open(save_trk_file, 'w')
      dets = seq_dets[seq_dets[:,0]==frame,7:14]
      ori_array = seq_dets[seq_dets[:,0]==frame,-1].reshape((-1, 1))
      other_array = seq_dets[seq_dets[:,0]==frame,1:7]
      additional_info = np.concatenate((ori_array, other_array), axis=1)
      consider_flag = (other_array[:,-1] > args.dthresh)
      dets_all = {'dets': dets[consider_flag], 'info': additional_info[consider_flag]}
      total_frames += 1
      start_time = time.time()
      trackers = mot_tracker.update(dets_all,center_kittif)
      cycle_time = time.time() - start_time
      total_time += cycle_time
      for (d,uid,tracked) in trackers:
        #print(d.dtype,d.shape)
        bbox3d_tmp = d[0:7]
        id_tmp = d[7]
        ori_tmp = d[8]
        type_tmp = det_id2str[d[9]]
        bbox2d_tmp_trk = d[10:14]
        conf_tmp = d[14]
        
        # convert kitti to argo
        x,y,z = bbox3d_tmp[5], -bbox3d_tmp[3], -bbox3d_tmp[4] #+ 1.78 + bbox3d_tmp[0]/2
        roi_pts = city_to_egovehicle_se3.inverse_transform_point_cloud(np.array([[x,y,z]]))  # put into city coords
        x,y,z = roi_pts[0]
        kitti_x,kitti_y,kitti_z = -y,1.73-z,x
        z += bbox3d_tmp[0]/2
        r_y = -np.pi/2 - bbox3d_tmp[6]
        
        if r_y >= np.pi: r_y -= np.pi * 2    
        if r_y < -np.pi: r_y+= np.pi * 2
            
        quat = euler_to_quaternion((r_y,0.0,0.0)) #q_x,qy,q_z,qw
        
        track = {}
        
        track["center"] = {
                    "x": x,
                    "y": y,
                    "z": z,
                }
        track["rotation"] = {
            "x": quat[0],
            "y": quat[1],
            "z": quat[2],
            "w": quat[3],
        }
        track["length"] = bbox3d_tmp[2]
        track["width"]  = bbox3d_tmp[1]
        track["height"] = bbox3d_tmp[0]
        track["occlusion"] = 0
        track["tracked"] = tracked
        track["timestamp"] = lidar_timestamp
        track["label_class"] = args.cat
        track["track_label_uuid"] = uid
        
        track_list.append(track)
        
        str_to_srite = '%s -1 -1 %f %f %f %f %f %f %f %f %f %f %f %f %f %d\n' % (type_tmp, ori_tmp,
          bbox2d_tmp_trk[0], bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3], 
          bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2], kitti_x, kitti_y, kitti_z, bbox3d_tmp[6], 
          conf_tmp, id_tmp)
        save_trk_file.write(str_to_srite)

        str_to_srite = '%d %d %s 0 0 %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (frame, id_tmp, 
          type_tmp, ori_tmp, bbox2d_tmp_trk[0], bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3], 
          bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2], kitti_x, kitti_y, kitti_z, bbox3d_tmp[6], 
          conf_tmp)
        eval_file.write(str_to_srite)

      with open(
      os.path.join(
      "argo_results", argo_save_name, seq_name,"per_sweep_annotations_amodal","tracked_object_labels_%s.json" % (lidar_timestamp)
      ),
      "w",
      ) as outfile:
        json.dump(track_list, outfile, indent=4)
        
        
      save_trk_file.close()

    eval_file.close()
      
  print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))
