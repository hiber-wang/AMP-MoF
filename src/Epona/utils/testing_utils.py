import numpy as np
import imageio
import os
import cv2
import matplotlib.pyplot as plt

def create_video(args, imgs, video_save_path, border_size = 10):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 
    fps = 2  
    condition_frames = args.condition_frames
    
    if not os.path.exists(video_save_path):
        os.makedirs(video_save_path)
    _, _, h, w = imgs[0].shape
    video_writer = cv2.VideoWriter(os.path.join(video_save_path, 'video.mp4'), fourcc, fps, (w+2*border_size, h+2*border_size))

    for j, image_file in enumerate(imgs):
        img = (image_file[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')[:,:,::-1]
        if j < condition_frames:
            bordered_img = add_border(img, border_size=border_size)
        else:
            bordered_img = add_border(img, border_size=border_size, value=[0, 0, 255])
        cv2.imwrite(os.path.join(video_save_path, '%d.png'%(j)), bordered_img)
        video_writer.write(bordered_img)

    video_writer.release()

def create_mp4(args, imgs, video_save_path, border_size = 10, fps=2):
    condition_frames = args.condition_frames
    print("save fps as ", fps)
    
    if not os.path.exists(video_save_path):
        os.makedirs(video_save_path)
    _, _, h, w = imgs[0].shape

    # video_writer = cv2.VideoWriter(os.path.join(video_save_path, 'video.mp4'), fourcc, fps, (w+2*border_size, h+2*border_size))
    with imageio.get_writer(os.path.join(video_save_path, 'video.mp4'), mode='I', fps=fps) as writer:
        for j, image_file in enumerate(imgs):
            img = (image_file[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')[:,:,::-1]
            if j < condition_frames:
                bordered_img = add_border(img, border_size=border_size)
            else:
                bordered_img = add_border(img, border_size=border_size, value=[0, 0, 255])
            # cv2.imwrite(os.path.join(video_save_path, '%d.png'%(j)), bordered_img)
            writer.append_data(bordered_img[:, :, ::-1])


def create_gif(args, imgs, video_save_path, border_size = 10, fps=2):
    images = []
    
    condition_frames = args.condition_frames
    
    if not os.path.exists(video_save_path):
        os.makedirs(video_save_path)
    _, _, h, w = imgs[0].shape

    # video_writer = cv2.VideoWriter(os.path.join(video_save_path, 'video.mp4'), fourcc, fps, (w+2*border_size, h+2*border_size))
    for j, image_file in enumerate(imgs):
        img = (image_file[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')[:,:,::-1]
        if j < condition_frames:
            bordered_img = add_border(img, border_size=border_size)
        else:
            bordered_img = add_border(img, border_size=border_size, value=[0, 0, 255])
        cv2.imwrite(os.path.join(video_save_path, '%d.png'%(j)), bordered_img)
        images.append(bordered_img[:, :, ::-1])

    imageio.mimsave(os.path.join(video_save_path, 'vis.gif'), images, fps=fps)


def add_border(img, border_size = 10, value=[255, 255, 255]):
    bordered_img = cv2.copyMakeBorder(img, 
                                      top=border_size, 
                                      bottom=border_size, 
                                      left=border_size, 
                                      right=border_size, 
                                      borderType=cv2.BORDER_CONSTANT,
                                      value=value)
    return bordered_img

def create_mp4_imgs(args, imgs, video_save_path, border_size = 10, fps=2, name=""):
    condition_frames = args.condition_frames
    print("save fps as ", fps)

    if not os.path.exists(video_save_path):
        os.makedirs(video_save_path)

    with imageio.get_writer(os.path.join(video_save_path, 'video'+name+'.mp4'), mode='I', fps=fps) as writer:
        for j, image_file in enumerate(imgs):
            if j < condition_frames:
                bordered_img = add_border(image_file, border_size=border_size)
            else:
                bordered_img = add_border(image_file, border_size=border_size, value=[0, 0, 255])
            # cv2.imwrite(os.path.join(video_save_path, '%d.png'%(j)), bordered_img)
            writer.append_data(bordered_img[:, :, ::-1])

def set_text(image, pose):
    # pose: string
    number = pose
    image = np.ascontiguousarray(image, dtype=np.uint8)

    height, width, _ = image.shape

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 0, 255) 
    thickness = 1

    (text_width, text_height), baseline = cv2.getTextSize(number, font, font_scale, thickness)

    x = width - text_width - 10  # 10 是边距
    y = text_height + 10  # 10 是边距
    print(number)

    # 在图片上写入数字
    image = cv2.putText(image, number, (x, y), font, font_scale, color, thickness)
    return image

def plot_trajectory(predict_traj, gt_traj, video_save_path, num):
    N = predict_traj.shape[0]
    fig, axs = plt.subplots(3, 1, figsize=(6, 6))
    labels = ['pose x', 'pose y', 'yaw']
    
    for i in range(3):
        axs[i].plot(range(N), predict_traj[:, i], label='predict', marker='o', markersize=4)
        axs[i].plot(range(N), gt_traj[:, i], label='gt', marker='o', markersize=4)
        
        for j in range(N):
            axs[i].annotate(f'{predict_traj[j, i]:.2f}', (j, predict_traj[j, i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, color='blue')
            axs[i].annotate(f'{gt_traj[j, i]:.2f}', (j, gt_traj[j, i]), textcoords="offset points", xytext=(0,-10), ha='center', fontsize=8, color='red')
        
        axs[i].set_xlabel('time')
        axs[i].set_ylabel(labels[i])
        axs[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(video_save_path, f'predict_traj_{num}.png'))
    plt.close()
