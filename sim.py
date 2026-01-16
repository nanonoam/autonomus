import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class RaceCarSimulator:
    def __init__(self, dt=0.1):
        self.dt = dt
        self.time = 0
        self.path_t = 0 
        
        # nois
        self.gps_noise_std = 0.5
        self.imu_accel_noise = 0.1
        self.imu_gyro_noise = 0.05
        self.perception_dist_noise = 0.2
        self.perception_angle_noise = 0.05
        
        # map for real
        self.ground_truth_map = [] 
        
        
        self.blue_cones = None
        self.yellow_cones = None
        

        self.generate_track()
        

        self.x = 20.0 
        self.y = 0.0
        self.theta = np.pi / 2 
        self.v = 5.0 

    def generate_track(self):
        """
        יוצר את רשימת הקונוסים המלאה ושומר אותה גם כמערכי Numpy לציור מהיר
        """
        a_inner, b_inner = 18, 8   # מסלול כחול
        a_outer, b_outer = 22, 12  # מסלול צהוב
        
        num_cones = 40 
        t_steps = np.linspace(0, 2*np.pi, num_cones, endpoint=False)
        
        cone_id_counter = 0
        blue_list = []
        yellow_list = []
        
        # blue
        for t in t_steps:
            cx = a_inner * np.cos(t)
            cy = b_inner * np.sin(t)
            cone = {'id': cone_id_counter, 'x': cx, 'y': cy, 'color': 'blue'}
            self.ground_truth_map.append(cone)
            blue_list.append([cx, cy])
            cone_id_counter += 1
            
        # and i was all_____
        for t in t_steps:
            cx = a_outer * np.cos(t)
            cy = b_outer * np.sin(t)
            cone = {'id': cone_id_counter, 'x': cx, 'y': cy, 'color': 'yellow'}
            self.ground_truth_map.append(cone)
            yellow_list.append([cx, cy])
            cone_id_counter += 1
            
        # make this to array to comper later
        self.blue_cones = np.array(blue_list)
        self.yellow_cones = np.array(yellow_list)

    def get_ground_truth_map(self):
        return self.ground_truth_map

    def step_physics(self):
        """ מזיזה את הרכב בדיוק לאורך המסלול האליפטי (Ground Truth) """
        # derivitiv of fild
        dx_dt = -20 * np.sin(self.path_t)
        dy_dt = 10 * np.cos(self.path_t)
        
        # calc speeds
        dist_step = self.v * self.dt
        radius_term = np.sqrt(dx_dt**2 + dy_dt**2)
        if radius_term == 0: radius_term = 0.001 # מניעת חלוקה באפס
        
        delta_t_param = dist_step / radius_term
        
        # place
        self.path_t += delta_t_param
        self.path_t %= (2 * np.pi)
        
        old_theta = self.theta
        
        # real state
        self.x = 20 * np.cos(self.path_t)
        self.y = 10 * np.sin(self.path_t)
        
        # angle
        new_dx = -20 * np.sin(self.path_t)
        new_dy = 10 * np.cos(self.path_t)
        self.theta = np.arctan2(new_dy, new_dx)
        
        
        angle_diff = (self.theta - old_theta + np.pi) % (2*np.pi) - np.pi
        real_omega = angle_diff / self.dt
        
        self.time += self.dt
        return real_omega

    def get_sensor_readings(self, true_omega):
        """ החזרת נתונים רועשים ל-EKF """
        # GPS
        gps = np.array([self.x + np.random.normal(0, self.gps_noise_std),
                        self.y + np.random.normal(0, self.gps_noise_std)])
        
        # IMU
        imu = {'accel_x': 0.0 + np.random.normal(0, self.imu_accel_noise),
               'gyro_z': true_omega + np.random.normal(0, self.imu_gyro_noise)}
        
        # Perception (Cones)
        visible_cones = []
        
        for cone in self.ground_truth_map:
            # vector to cone
            dx = cone['x'] - self.x
            dy = cone['y'] - self.y
            
            # rotate to mache the feeld
            lx = dx * np.cos(-self.theta) - dy * np.sin(-self.theta)
            ly = dx * np.sin(-self.theta) + dy * np.cos(-self.theta)
            
            dist = np.sqrt(lx**2 + ly**2)
            
            # only see the front(we are not a math teacher)
            if 0 < lx < 15 and abs(ly) < 10:
                noisy_dist = dist + np.random.normal(0, self.perception_dist_noise)
                scale = noisy_dist / dist
                visible_cones.append({
                    'id': cone['id'],
                    'x': lx * scale, 
                    'y': ly * scale,
                    'color': cone['color'],
                    'dist': noisy_dist
                })
        
        visible_cones.sort(key=lambda c: c['dist'])
        return gps, imu, visible_cones[:6] 

    #no clue how this works, i let gpt make the drawings
    def render(self, ax, gps_data, visible_cones):
        """ ציור הגרפיקה """
        ax.clear()
        
        # 1. ציור מסלול אמיתי (כחול וצהוב) - משתמש במערכים שיצרנו ב-generate_track
        if self.blue_cones is not None:
            ax.scatter(self.blue_cones[:,0], self.blue_cones[:,1], c='blue', s=10, label='True Blue')
        if self.yellow_cones is not None:
            ax.scatter(self.yellow_cones[:,0], self.yellow_cones[:,1], c='gold', s=10, label='True Yellow')
        
        # 2. ציור הרכב האמיתי
        car_circle = patches.Circle((self.x, self.y), radius=0.8, color='red', label='True Car', alpha=0.5)
        arrow = patches.Arrow(self.x, self.y, 2*np.cos(self.theta), 2*np.sin(self.theta), width=0.5, color='black')
        ax.add_patch(car_circle)
        ax.add_patch(arrow)
        
        # 3. ציור GPS
        ax.scatter(gps_data[0], gps_data[1], c='green', marker='x', s=100, label='GPS Raw')
        
        # 4. ויזואליזציה של מה שהרובוט רואה (קו ירוק לקונוסים שמזוהים כרגע)
        # כדי לצייר, צריך להמיר חזרה לגלובלי רק בשביל הגרף
        for vc in visible_cones:
            # המרה הפוכה (רק לויזואליזציה): Local -> Global
            # x_global = x_robot + (x_local * cos(theta) - y_local * sin(theta))
            # y_global = y_robot + (x_local * sin(theta) + y_local * cos(theta))
            gx = self.x + (vc['x'] * np.cos(self.theta) - vc['y'] * np.sin(self.theta))
            gy = self.y + (vc['x'] * np.sin(self.theta) + vc['y'] * np.cos(self.theta))
            ax.plot([self.x, gx], [self.y, gy], 'k-', alpha=0.2) # קו דק שמראה זיהוי
            
        ax.set_xlim(-30, 30)
        ax.set_ylim(-20, 20)
        ax.legend(loc='upper right')
        ax.set_title(f"Sim Time: {self.time:.1f}s | Speed: {self.v} m/s")
        plt.pause(0.01)

#main carictar
if __name__ == "__main__":
    sim = RaceCarSimulator(dt=0.1)
    
    # save map for later
    true_map = sim.get_ground_truth_map()
    print(f"Ground Truth Map Generated: {len(true_map)} cones.")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    try:
        # for ever and ever(till deth do us part)
        while True: 

            true_omega = sim.step_physics()
            
            # get parameters
            gps, imu, cones = sim.get_sensor_readings(true_omega)
            
            # code go her
            

            
            sim.render(ax, gps, cones)
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
        plt.close()