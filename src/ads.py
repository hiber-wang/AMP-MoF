import csv
import math
import sys
import random
import os
from queue import Queue
from queue import Empty
import time

import os
import torch
torch.set_printoptions(sci_mode=False)
import random
import sys
import torchvision.transforms as transforms
from PIL import Image

from driving_model.model_car import Resnet101Steer, Resnet101Speed, Vgg16Steer, Vgg16Speed, EpochSpeed, EpochSteer

try:
    sys.path.append("/home/weizi/workspace/misbehavior_prediction/carla/PythonAPI/carla/dist/carla-0.9.10-py3.9-linux-x86_64.egg")
    import carla
except:
    print("Couldn't set Carla API path.")
    exit(-1)

collision_flag = False
    
def get_control(model_speed, model_steer, imgs, device):
    imgs = imgs.unsqueeze(0)
    imgs = imgs.type(torch.FloatTensor)
    imgs = imgs.to(device)
    
    model_speed.eval()
    speed = model_speed(imgs)

    model_steer.eval()
    steer = model_steer(imgs)
    return speed.item(), steer.item()



def sensor_callback(sensor_data, sensor_queue, sensor_name):
    global collision_flag
    if 'camera' in sensor_name:
        pass

    if 'collision' in sensor_name:
        print("collision")
        collision_flag = True
    if 'imu' in sensor_name:
        pass

    sensor_queue.put((sensor_data, sensor_name))


def run_ads(args):
    device = torch.device(args.device)
    img_transform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor()
            ])
    
    actor_list = []
    sensor_list = []
    global collision_flag
    runout_flag = False
    lane_invasion_flag = False
    try:
        # First of all, we need to create the client that will send the requests, assume port is 2000
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)

        # Retrieve the world that is currently running
        town_idx = random.choice([1, 2])
        world = client.load_world('Town0' + str(town_idx)) # you can also retrive another world by specifically defining

        blueprint_library = world.get_blueprint_library()

        # Set weather for your world
        if args.agent == "beahvior":
            weather = world.get_weather()
            weather.cloudiness = random.randint(0, 50)
            weather.precipitation = random.randint(0, 100)
            weather.precipitation_deposits = random.randint(0, 50)
            weather.fog_distance = random.randint(0, 100)
            weather.wetness = random.randint(0, 100)
            weather.wind_intensity = random.randint(0, 100)
            weather.fog_density = random.randint(0, 100)
            weather.sun_azimuth_angle = random.randint(0, 360)
            weather.sun_altitude_angle = random.randint(60, 90)
        else:
            weather = world.get_weather()
            weather.cloudiness = random.randint(0, 50)
            weather.precipitation = random.randint(0, 100)
            weather.precipitation_deposits = random.randint(0, 50)
            weather.fog_distance = random.randint(0, 100)
            weather.wetness = random.randint(0, 100)
            weather.wind_intensity = random.randint(0, 100)
            weather.fog_density = random.randint(0, 100)
            weather.sun_azimuth_angle = random.randint(0, 360)
            weather.sun_altitude_angle = random.randint(40, 90)

        world.set_weather(weather)

        # set synchorinized mode
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

        # create sensor queue
        sensor_queue = Queue()



        # create the ego vehicle
        ego_vehicle_bp = blueprint_library.find('vehicle.mercedes-benz.coupe')
        ego_vehicle_bp.set_attribute('color', '0, 0, 0')
        # get a random valid occupation in the world
        transform = random.choice(world.get_map().get_spawn_points())
        # spawn the vehilce
        ego_vehicle = world.spawn_actor(ego_vehicle_bp, transform)
        # set the vehicle autopilot mode
        print(args.agent)
        if args.agent == "behavior":
            ego_vehicle.set_autopilot(True)
            if not args.normal:
                traffic_manager.ignore_vehicles_percentage(ego_vehicle, 100)
                traffic_manager.ignore_walkers_percentage(ego_vehicle, 100)
                traffic_manager.ignore_lights_percentage(ego_vehicle, 100)
            else:
                traffic_manager.set_global_distance_to_leading_vehicle(random.uniform(2.6, 3))
        elif args.agent == 'resnet101':
            model_speed = Resnet101Speed()
            model_speed.load_state_dict(torch.load(os.path.join(args.driving_model, args.agent, "speed.pt")))
            model_speed = model_speed.to(device)

            model_steer = Resnet101Steer()
            model_steer.load_state_dict(torch.load(os.path.join(args.driving_model, args.agent, "steer.pt")))
            model_steer = model_steer.to(device)

        elif args.agent == 'vgg16':
            model_speed = Vgg16Speed()
            model_speed.load_state_dict(torch.load(os.path.join(args.driving_model, args.agent, "speed.pt")))
            model_speed = model_speed.to(device)

            model_steer = Vgg16Steer()
            model_steer.load_state_dict(torch.load(os.path.join(args.driving_model, args.agent, "steer.pt")))
            model_steer = model_steer.to(device)
        
        elif args.agent == 'epoch':
            model_speed = EpochSpeed()
            model_speed.load_state_dict(torch.load(os.path.join(args.driving_model, args.agent, "speed.pt")))
            model_speed = model_speed.to(device)

            model_steer = EpochSteer()
            model_steer.load_state_dict(torch.load(os.path.join(args.driving_model, args.agent, "steer.pt")))
            model_steer = model_steer.to(device)
        # collect all actors to destroy when we quit the script
        actor_list.append(ego_vehicle)

        # create surrounding pedestrians
        walker_bp = blueprint_library.find("walker.pedestrian.0001")
        walker_controller_bp = blueprint_library.find('controller.ai.walker')
        
        spawned_points = []
        spawned_points.append(transform)
        for i in range(args.num_walkers):
            actor_sp = random.choice(world.get_map().get_spawn_points())
            actor_walker = world.try_spawn_actor(walker_bp, actor_sp)
            while actor_walker is None:
                 actor_sp = random.choice(world.get_map().get_spawn_points())
                 actor_walker = world.try_spawn_actor(walker_bp, actor_sp)
            controller_walker = world.spawn_actor(walker_controller_bp, actor_sp, actor_walker)
            actor_list.append(actor_walker)
            world.tick() # without this, walker vanishes
            controller_walker.start()
            controller_walker.go_to_location(random.choice(world.get_map().get_spawn_points()).location)
            spawned_points.append(actor_sp)
            actor_walker.set_simulate_physics(True)
        # create surrounding vehicles
        blueprints = world.get_blueprint_library().filter("vehicle.*")
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]

        for i in range(args.num_vehicles):
            blueprint = random.choice(blueprints)
            # blueprint = blueprint_library.find('vehicle.audi.tt')
            npc_transform = random.choice(world.get_map().get_spawn_points())
            while npc_transform in spawned_points:
                npc_transform = random.choice(world.get_map().get_spawn_points())
    
            npc_vehicle = world.spawn_actor(blueprint, npc_transform)
            npc_vehicle.set_autopilot(True)
            actor_list.append(npc_vehicle)
            spawned_points.append(npc_transform)

        if args.record:
            output_time = time.time()
            output_path = os.path.join(args.output, args.agent, str(output_time))
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            csv_path  = os.path.join(args.output, args.agent, str(output_time)+".csv")
            if not os.path.exists(csv_path):
                with open(csv_path,'w') as f:
                    csv_write = csv.writer(f)
                    csv_head = ["frame","speed", "accelerometer_x", "accelerometer_y", "accelerometer_z", "gyroscope_x", "gyroscope_y", "gyroscope_z", "throttle", "steer", "brake"]
                    csv_write.writerow(csv_head)

        # add a camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')

        camera_bp.set_attribute("image_size_x", "704")
        camera_bp.set_attribute("image_size_y", "384")

        camera_front_transform = carla.Transform(carla.Location(x=2.5, z=1.0))
        camera_front = world.spawn_actor(camera_bp, camera_front_transform, attach_to=ego_vehicle)
        # set the callback function
        camera_front.listen(lambda image: sensor_callback(image, sensor_queue, "camera_front"))
        sensor_list.append(camera_front)



        # add a collision detector
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision = world.spawn_actor(collision_bp, carla.Transform(), attach_to=ego_vehicle)
        collision.listen(lambda collision: sensor_callback(collision, sensor_queue, "collision"))


        # add an IMU measure
        imu_bp = blueprint_library.find('sensor.other.imu')
        imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=ego_vehicle)
        imu.listen(lambda imu: sensor_callback(imu, sensor_queue, "imu"))
        sensor_list.append(imu)
        
        tick = 0
        speed = 0
        steer = 0

        lane_invasion_cnt = 0

        while True:
            if collision_flag or runout_flag or lane_invasion_flag:
                exit(-1)

            if args.record:
                capture_num = os.listdir(output_path)
                if args.agent != "behavior" and len(capture_num) >= 300:
                    exit(-1)
                if args.normal and len(capture_num) >= args.time:
                    exit(-1)
                if not args.normal and len(capture_num) == 50:
                    ego_vehicle.set_autopilot(False)
                    vehicle_control = carla.VehicleControl(throttle=max(0.3, random.random()), steer=random.random() * 2 - 1)
                    ego_vehicle.apply_control(vehicle_control)

            world.tick()
            # set the sectator to follow the ego vehicle
            spectator = world.get_spectator()
            transform = ego_vehicle.get_transform()
            location = ego_vehicle.get_location()
            wp = world.get_map().get_waypoint(location, project_to_road=True)
            wp_location = wp.transform.location

            if math.sqrt((wp_location.x - location.x)**2 + (wp_location.y - location.y)**2 + (wp_location.z - location.z)**2 ) >= 2:
                print("not in driving area")
                runout_flag = True
            spectator.set_transform(carla.Transform(transform.location + carla.Location(z=2), transform.rotation))

            velocity = ego_vehicle.get_velocity()
            current_speed = math.sqrt((velocity.x) ** 2 + (velocity.y)**2)

            wheels = ego_vehicle.get_physics_control().wheels
            wheel_fl = wheels[0]
            wheel_fr = wheels[1]
            wheel_fl_location = wheel_fl.position / 100
            wheel_fr_location  = wheel_fr.position / 100

            wheel_fr_waypoint = world.get_map().get_waypoint(wheel_fr_location)
            wheel_fl_waypoint = world.get_map().get_waypoint(wheel_fl_location)

            if args.agent == "behavior":
                if not wheel_fr_waypoint.is_junction and wheel_fr_waypoint.lane_id != wheel_fl_waypoint.lane_id:
                    lane_invasion_cnt += 1
                    if lane_invasion_cnt > 5:
                        lane_invasion_flag = True
                        print("lane_invasion")
                else:
                    lane_invasion_cnt = 0

            if args.agent != "behavior":
                if speed > current_speed:
                    ego_vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=steer))
                elif speed < current_speed:
                    ego_vehicle.apply_control(carla.VehicleControl(brake=1, steer=steer))
                else:
                    ego_vehicle.apply_control(carla.VehicleControl(steer=steer))

            # As the queue is blocking, we will wait in the queue.get() methods
            # until all the information is processed and we continue with the next frame.
            try:
                s_data_list = []
                for i in range(0, len(sensor_list)):
                    s_data, s_name = sensor_queue.get(True, 2.0)
                    s_data_list.append({"name": s_name, "data": s_data})

            except Empty:
                print("   Some of the sensor information is missed")
            if tick % 2 == 0:
                for s_data in s_data_list:
                    if s_data["name"] == "camera_front":
                        if args.record:
                            s_data["data"].save_to_disk(os.path.join(output_path, str(s_data["data"].frame) + '.png'))
                            imgs = Image.open(os.path.join(output_path, str(s_data["data"].frame) + '.png')).convert("RGB")
                        else:
                            s_data["data"].save_to_disk(os.path.join(args.output,'0.png'))
                            imgs = Image.open(os.path.join(args.output, '0.png')).convert("RGB")
                        imgs = img_transform(imgs)
                        if args.agent != "behavior":
                            speed, steer = get_control(model_speed, model_steer, imgs, device)
                            print(speed, steer)

                        if args.agent != "behavior":
                            speed, steer = get_control(model_speed, model_steer, imgs, device)
                    if s_data["name"] == "imu":
                        ego_velocity = ego_vehicle.get_velocity()
                        speed = math.sqrt((ego_velocity.x)**2 + (ego_velocity.y) ** 2 + (ego_velocity.z)**2)
                        accelerometer_x = s_data["data"].accelerometer.x
                        accelerometer_y = s_data["data"].accelerometer.y
                        accelerometer_z = s_data["data"].accelerometer.z
                        gyroscope_x = s_data["data"].gyroscope.x
                        gyroscope_y = s_data["data"].gyroscope.y
                        gyroscope_z = s_data["data"].gyroscope.z
                        control = ego_vehicle.get_control()
                        throttle = control.throttle
                        steer = control.steer
                        brake = control.brake
                        if args.record:
                            with open(csv_path, "a+") as f:
                                csv_write = csv.writer(f)
                                csv_write.writerow([s_data["data"].frame, speed, accelerometer_x, accelerometer_y, accelerometer_z, gyroscope_x, gyroscope_y, gyroscope_z, throttle, steer, brake])
            tick += 1
    finally:
        if args.record:
            record_flag = 0
            if collision_flag:
                record_flag = 1
            elif runout_flag or lane_invasion_flag:
                record_flag = 2
            else:
                pass
            filename= os.path.join(args.output, args.agent, "record.txt")
            # 追加写入内容到文件
            with open(filename, "a") as file:
                file.write(str(output_time) + " " + str(record_flag) + "\n")
            
        world.apply_settings(original_settings)
        print('destroying actors')
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        for sensor in sensor_list:
            sensor.destroy()
        collision.destroy()

if __name__ == '__main__':
    try:
        run_ads()
    except KeyboardInterrupt:
        print(' - Exited by user.')
