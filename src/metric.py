from collections import deque
import os
import pickle
import matplotlib.pyplot as plt

def draw_risk(scores, folder, R, max_len):
    
    max_deque = deque(maxlen=max_len)
    folder_scores = scores[folder]
    score_trend = []
    for folder_score in folder_scores:
        image, score, running_time = folder_score
        max_deque.append(score)
        score_trend.append(sum(max_deque) / len(max_deque))
    
    range_set = 0
    plt.plot(score_trend[range_set:])
    begin = len(folder_scores) - 1 - 5 - range_set
    end = len(folder_scores) - 1 - range_set
    x = [0, end]
    y = [R, R]
    plt.plot(x, y, color="red", linestyle="--", linewidth=0.5)

    x = [begin, end]
    y = [R, R]
    plt.plot(x, y, color="red")

    plt.xticks([])

    # plt.scatter([len(folder_scores)-1-60-range_set], [1])
    plt.show()

def oracle_for_warning(scores, folder, R, max_len):
    max_deque = deque(maxlen=max_len)
    warning_frames = []
    scores_frames = []
    folder_scores = scores[folder]
    for folder_score in folder_scores:
        image, score, running_time = folder_score
        max_deque.append(score)
        if sum(max_deque) / len(max_deque) > R:
            warning_frames.append(image)
            scores_frames.append(sum(max_deque) / len(max_deque))
        # if min(max_deque) > R:
        #     warning_frames.append(image)
        #     scores_frames.append(min(max_deque))
    return warning_frames, scores_frames


if __name__ == "__main__":
    oracle = "Atten80.5"
    agent = "epoch"
    abnormal_end =5
    abnormal_start = 60
    print(oracle, agent, abnormal_end)
    R = 1.3752680898260206e-07
    # R = 0.023
    # R = 0.05
    # R = 0.01
    # R = 0.002
    if oracle == "Atten" or oracle=="Atten-8":
        R = 1
    elif oracle == "Atten80.75":
        R = 1.8
    elif oracle == "Atten80.0":
        R = 0.8
    elif oracle == "Atten80.25":
        R = 1
    elif oracle == "Atten80.5":
        R = 0.87
    elif oracle == "Atten11":
        R = 0.1
    elif oracle == "Atten_ae81":
        R = 92.92009028256153
    elif oracle == "Atten161":
        R = 0.88
    elif oracle == "SelfOracle":
        R = 0.023
    max_len =1
    tp = 0
    tn = 0
    fp = 0
    fn  = 0

    task = 1
    
    
    scores_path = os.path.join("output", agent, oracle+".pkl")
    with open(scores_path, "rb") as f:
        scores = pickle.load(f)
    if task == 0:
        draw_risk(scores, os.path.join("output", agent, "1715684880.3912163"), R, max_len) #  1715684880.3912163


    elif task == 1:
        folders_path = os.path.join("output", agent, "record.txt")
        with open(folders_path, "r") as f:
            folders = f.readlines()
            for line in folders:
                line = line.strip('\n')
                data = line.split(" ")
                folder, label = data[0], data[1]
                folder_path = os.path.join("output", agent, folder)
                frames = os.listdir(folder_path)
                frames = list(frames)
                frames.sort(key=lambda x: int(x[:-4]))
                if label == "0":
                    warning_frames, scores_frames = oracle_for_warning(scores, folder_path, R, max_len)
                    frame_lists = [frames[i:i+30] for i in range(0, len(frames), 30)]
                    for frame_list in frame_lists:
                        flag = False
                        for warning_frame in warning_frames:
                            if warning_frame in frame_list:
                                fp += 1
                                flag = True
                                # print(folder, warning_frames)
                                break
                        if not flag:
                            tn += 1
                else:
                    warning_frames, _ = oracle_for_warning(scores, folder_path, R, max_len)
                    # 最后5帧作为反应时间，反应时间帧之前的60帧作为察觉时间
                    reaction_frames = frames[-1*abnormal_end:]
                    abnormal_frames = frames[-1*abnormal_start:-1*abnormal_end]
                    normal_frame_lists = [frames[i:i+30] for i in range(0, len(frames[:-1*abnormal_start]), 30)]
                    for normal_frame_list in normal_frame_lists:
                        flag = False
                        for warning_frame in warning_frames:
                            if warning_frame in normal_frame_list:
                                fp += 1
                                flag = True
                                # print(folder, warning_frame)
                                break
                        if not flag:
                            tn += 1
                    
                    flag = False
                    for warning_frame in warning_frames:   
                        if warning_frame in abnormal_frames:
                            tp += 1
                            flag = True
                            # print(folder, label)
                            break
                    if not flag:
                        # print(folder)
                        fn += 1
                precision = tp / (tp+fp) if tp + fp > 0 else 0
                recall = tp / (tp + fn) if tp + fn > 0 else 0
                F3 = 10*precision*recall / (9*precision+recall) if precision + recall > 0 else 0
                # F3 = 2 * precision * recall / (precision + recall)
            print(precision, recall, F3)
            print("正->正", tp, "负->负", tn, "负->正", fp, "正->负", fn)