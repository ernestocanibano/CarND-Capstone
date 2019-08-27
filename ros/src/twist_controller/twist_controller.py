from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
MIN_SPEED = 0.1


class Controller(object):
    def __init__(self, *args, **kwargs):
        # TODO: Implement
        self.yaw_controller = YawController(kwargs['wheel_base'], kwargs['steer_ratio'], MIN_SPEED, kwargs['max_lat_accel'], kwargs['max_steer_angle'])
            
        # Create PID to control throttle
        mn=0.0 # minimum throttle value
        mx=0.8 # maximum throttle value
        self.throttle_controller = PID(kp=0.8, ki=0.001, kd=0.2, mn=mn, mx=mx) #ki=0.15, kd=0.0001
	
        # Create PID to control steering angle
        mn=-kwargs['max_steer_angle'] # minimum steering angle
        mx=kwargs['max_steer_angle']  # maximum steering angle
        self.steer_controller = PID(kp=1.4, ki=0.000001, kd=0.2, mn=mn, mx=mx)

        # Set low pass filter parameters: sample_time=1/50hz, cutoff_frecuency=tau
        tau = 0.5 # 1/(2pi*tau) = cutoff frequency
        ts = 0.02 # Sample time = 1 / 50hz
        self.vel_lpf = LowPassFilter(tau, ts)

        self.vehicle_mass = kwargs['vehicle_mass']
        self.fuel_capacity = kwargs['fuel_capacity']
        self.brake_deadband = kwargs['brake_deadband']
        self.decel_limit = kwargs['decel_limit']
        self.accel_limit = kwargs['accel_limit']
        self.wheel_radius = kwargs['wheel_radius']

        # Add the fuel mass to the vehicle mass
        self.vehicle_mass = self.vehicle_mass + self.fuel_capacity*GAS_DENSITY
       
        self.last_time = rospy.get_time()

    def control(self, current_vel, dbw_enabled, target_linear_vel, target_angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer

        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0.0, 0.0, 0.0
        
        # Filter current speed of the car
        #not_filtered_vel = current_vel
        current_vel = self.vel_lpf.filt(current_vel)
        #rospy.loginfo("dbw_node: LowPassFilter (not_filtered_speed={0}, filtered_speed={1})".format(not_filtered_vel, current_vel))
        
        current_time = rospy.get_time()
        dt = current_time - self.last_time
        self.last_time = current_time
       
        # Calculate steering angle using yaw controller and pid controller
        steering = self.yaw_controller.get_steering(target_linear_vel, target_angular_vel, current_vel)
        steering = self.steer_controller.step(steering, dt)

        # Calculate throttle using pid controller
        vel_error = target_linear_vel - current_vel
        self.last_vel = current_vel
        throttle = self.throttle_controller.step(vel_error, dt)
        
        brake = 0

        if target_linear_vel == 0.0 and current_vel < 0.1: # Stop the car an activate brake
            throttle = 0
            brake = 700 #N*m - to hold the car in place if we are stopped at light. Acceleration ~ 1m/s^2
        elif throttle < 0.1 and vel_error < 0: # Decrease speed of the car
            throttle = 0
            decel = max(vel_error/dt, self.decel_limit) #tranform speed in acceleration
            brake = abs(decel)*self.vehicle_mass*self.wheel_radius #Torque N*m
               
        return throttle, brake, steering
