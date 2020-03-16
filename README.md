This code modifies AB3DMOT code https://github.com/xinshuoweng/AB3DMOT (main.py file ) for use with Argoverse dataset.

Usage (parameter values used for final model) - 

python main.py car_3d_det_val --dthresh=.25 --mal=2 --mah=2 --min_hits=3 --ithresh=0.1 --cat="VEHICLE" --set="val" --keep_age=3 --svel=0.3 --sdis=2                              

For evaluation (on val set ) - 

python eval_tracks.py --fname=val_t25.0_a22_h3_ioudot10.0_ka3_v0.3_d2.0 --set=val

