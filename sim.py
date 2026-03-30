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
        self.v = 20

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
            
        # and it was all_____
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
        """ מזיזה את הרכב ומוסיפה דינמיקה של תאוצה ובלימה """
        # 1. יצירת תאוצה משתנה (האצה ובלימה) - סינוס שמושפע מהזמן + קצת רעש רנדומלי
        self.a = 2.0 * np.sin(self.time) + np.random.normal(0, 0.5)
        
        # 2. עדכון המהירות לפי התאוצה האמיתית (v = v + a*dt)
        self.v += self.a * self.dt
        
        # נגביל את המהירות כדי שהרכב לא ייסע אחורה או יטוס לחלל
        self.v = np.clip(self.v, 2.0, 15.0) 

        # --- שאר הקוד המקורי של התנועה נשאר זהה ---
        dx_dt = -20 * np.sin(self.path_t)
        dy_dt = 10 * np.cos(self.path_t)
        
        dist_step = self.v * self.dt
        radius_term = np.sqrt(dx_dt**2 + dy_dt**2)
        if radius_term == 0: radius_term = 0.001 
        
        delta_t_param = dist_step / radius_term
        self.path_t += delta_t_param
        self.path_t %= (2 * np.pi)
        
        old_theta = self.theta
        self.x = 20 * np.cos(self.path_t)
        self.y = 10 * np.sin(self.path_t)
        
        new_dx = -20 * np.sin(self.path_t)
        new_dy = 10 * np.cos(self.path_t)
        self.theta = np.arctan2(new_dy, new_dx)
        
        angle_diff = (self.theta - old_theta + np.pi) % (2*np.pi) - np.pi
        real_omega = angle_diff / self.dt
        
        self.time += self.dt
        return real_omega

    def get_sensor_readings(self, true_omega):
        """ החזרת נתונים מהחיישנים כולל האינקודר החדש """
        gps = np.array([self.x + np.random.normal(0, self.gps_noise_std),
                        self.y + np.random.normal(0, self.gps_noise_std)])
        
        # ה-IMU עכשיו מקבל את התאוצה האמיתית שחישבנו
        imu = {'accel_x': self.a + np.random.normal(0, self.imu_accel_noise),
               'gyro_z': true_omega + np.random.normal(0, self.imu_gyro_noise)}
        
        # --- החיישן החדש: אינקודר (מודד מהירות) ---
        encoder_v = self.v + np.random.normal(0, 0.2) # שונות רעש של 0.2
        
        visible_cones = []
        for cone in self.ground_truth_map:
            dx = cone['x'] - self.x
            dy = cone['y'] - self.y
            lx = dx * np.cos(-self.theta) - dy * np.sin(-self.theta)
            ly = dx * np.sin(-self.theta) + dy * np.cos(-self.theta)
            dist = np.sqrt(lx**2 + ly**2)
            if 0 < lx < 15 and abs(ly) < 10:
                noisy_dist = dist + np.random.normal(0, self.perception_dist_noise)
                scale = noisy_dist / dist
                visible_cones.append({
                    'id': cone['id'], 'x': lx * scale, 'y': ly * scale,
                    'color': cone['color'], 'dist': noisy_dist
                })
        
        visible_cones.sort(key=lambda c: c['dist'])
        
        # נחזיר גם את האינקודר
        return gps, imu, encoder_v, visible_cones[:6]

    #no clue how this works, i let gpt make the drawings(and the phosics mosly)
    def render(self, ax, gps_data, visible_cones, ekf_state=None):
        """ ציור הגרפיקה """
        ax.clear()
        
        # 1. ציור מסלול אמיתי (כחול וצהוב) - משתמש במערכים שיצרנו ב-generate_track
        if self.blue_cones is not None:
            ax.scatter(self.blue_cones[:,0], self.blue_cones[:,1], c='blue', s=10, label='FR Blue')
        if self.yellow_cones is not None:
            ax.scatter(self.yellow_cones[:,0], self.yellow_cones[:,1], c='gold', s=10, label='FR Yellow')
        
        # 2. ציור הרכב האמיתי
        car_circle = patches.Circle((self.x, self.y), radius=0.8, color='red', label='FR Car', alpha=0.5)
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
        
        if ekf_state is not None:
            ekf_x = ekf_state[0, 0]
            ekf_y = ekf_state[1, 0]
            ax.scatter(ekf_x, ekf_y, c='purple', marker='*', s=200, label='EKF Estimate', zorder=5)

        ax.set_xlim(-30, 30)
        ax.set_ylim(-20, 20)
        ax.legend(loc='upper right')
        ax.set_title(f"Sim Time: {self.time:.1f}s | Speed: {self.v} m/s")
        plt.pause(0.01)


class EKF:
    def __init__(self):
        # מתחילים את ההערכה שלנו בדיוק - (לא בערך) איפה שהרכב מתחיל
        self.X = np.array([[20.0], 
                           [0.0], 
                           [np.pi/2], 
                           [20.0]])
        
        self.P = np.diag([1.0, 1.0, 0.1, 1.0])
        self.Q = np.diag([0.001, 0.001, 0.001, 0.01]) # how mach we mo trust the calc(IMU)

    def predict(self, dt, omega, a):
        x = self.X[0, 0]
        y = self.X[1, 0]
        theta = self.X[2, 0]
        v = self.X[3, 0]
        
        Xnew = x + v * np.cos(theta) * dt
        Ynew = y + v * np.sin(theta) * dt
        THETAnew = theta + omega * dt
        Vnew = v + a * dt
        
        self.X = np.array([[Xnew], [Ynew], [THETAnew], [Vnew]])
        
        J_F = np.array([[1, 0, -v * np.sin(theta) * dt, np.cos(theta) * dt],
                        [0, 1,  v * np.cos(theta) * dt, np.sin(theta) * dt],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        
        self.P = J_F @ self.P @ J_F.T + self.Q

    def update_gps(self, gps_x, gps_y):
        Z = np.array([[gps_x], 
                      [gps_y]])
        
        J_H = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]])
        
        expected_x = self.X[0, 0]
        expected_y = self.X[1, 0]
        Z_expected = np.array([[expected_x], 
                               [expected_y]])
        
        R = np.array([[0.25, 0],
                      [0, 0.25]]) # how mach we no trust the GPS
                      
        Y = Z - Z_expected
        S = J_H @ self.P @ J_H.T + R
        K = self.P @ J_H.T @ np.linalg.inv(S)
        
        self.X = self.X + K @ Y
        self.P = (np.eye(4) - K @ J_H) @ self.P

    def update_encoder(self, encoder_v):
        Z = np.array([[encoder_v]])
        
        J_H = np.array([[0, 0, 0, 1]])
        
        expected_v = self.X[3, 0]
        Z_expected = np.array([[expected_v]])
        
        R = np.array([[0.04]]) # how mach we no trust the encoder
                      
        Y = Z - Z_expected
        S = J_H @ self.P @ J_H.T + R
        K = self.P @ J_H.T @ np.linalg.inv(S)
        
        self.X = self.X + K @ Y
        self.P = (np.eye(4) - K @ J_H) @ self.P

# main character
if __name__ == "__main__":
    sim = RaceCarSimulator(dt=0.1)
    
    # save map for later
    true_map = sim.get_ground_truth_map()
    print(f"Ground Truth Map Generated: {len(true_map)} cones.")
    
    fig, ax = plt.subplots(figsize=(10, 6))

    ekf = EKF()
    try:
        # for ever and ever(till deth do us part)
        while True: 

            true_omega = sim.step_physics()
            
            # get parameters
            gps, imu, encoder_v, cones = sim.get_sensor_readings(true_omega)
            
            # add ekf
            
            # predict based on fizika
            ekf.predict(sim.dt, imu['gyro_z'], imu['accel_x'])
            
            # update based on GPS(very bad)
            ekf.update_gps(gps[0], gps[1])

            # update based on encoder (very good)
            ekf.update_encoder(encoder_v)
            
            # 3. ציור הגרפיקה על המסך
            if ekf.X is not None:
                sim.render(ax, gps, cones, ekf.X)
            else:
                sim.render(ax, gps, cones)
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
        plt.close()