"""Comparable motion metrics computed from timestamped wheel velocities."""
import numpy as np


def motion_metrics(times, wheels, wheel_radius=.027, axle_track=.104):
    times=np.asarray(times,dtype=float)
    wheels=np.asarray(wheels,dtype=float)
    if len(times)<2:
        return {}
    if wheels.shape!=(len(times),2) or np.any(np.diff(times)<=0):
        raise ValueError('Expected paired wheel samples at strictly increasing times')
    linear=wheel_radius*wheels.mean(axis=1)
    angular=wheel_radius*(wheels[:,1]-wheels[:,0])/axle_track
    turning=np.abs(angular)>.4
    intervals=np.diff(times)
    acceleration=np.diff(wheels,axis=0)/intervals[:,None]
    yaw_acceleration=np.diff(angular)/intervals
    yaw_jerk=np.diff(yaw_acceleration)/((intervals[1:]+intervals[:-1])/2)
    return dict(mean_forward_speed_m_s=float(linear.mean()),
                peak_forward_speed_m_s=float(linear.max()),
                peak_abs_yaw_rate_rad_s=float(np.abs(angular).max()),
                peak_wheel_acceleration_rad_s2=float(np.abs(acceleration).max()),
                rms_yaw_jerk_rad_s3=float(np.sqrt(np.mean(yaw_jerk**2))) if len(yaw_jerk) else None,
                moving_turn_fraction=float(np.mean(np.abs(linear[turning])>.05)) if np.any(turning) else None,
                stopped_fraction=float(np.mean(np.abs(linear)<.02)))
