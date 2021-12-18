import os
import glob
import re
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from itertools import chain
import cv2
from imutils.video import FileVideoStream
import time
import multiprocessing as mp
from functools import partial
import glob

##TODO:  Still need to finish this class so it will stop feeding the queue after its section has processed
class FileVideoStreamMP(FileVideoStream):
    def __init__(self, path, num_processes, group_number, **kwargs):
        super().__init__(path, **kwargs)
        self.num_processes = num_processes
        self.group_number = group_number
        self.frame_jump_unit = self.stream.get(cv2.CAP_PROP_FRAME_COUNT) // self.num_processes
        self.stream.set(cv2.CAP_PROP_POS_FRAMES, self.frame_jump_unit * self.group_number)
        self.proc_frames = 0

        # 160320: 1700, 180
        # 83512718053: 1600, 180
        # 170127: 2030, 180
        # 220120: 1600, 180
        #self.transform = lambda x: mask_rh_corner(x, 1600, 180)
        self.transform = lambda x: mask_rh_corner(x, 0.17, 0.16)

    def update(self):
        #print("DEBUG: In child class update method")
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                break

            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stopped = True

                # if there are transforms to be done, might as well
                # do them on producer thread before handing back to
                # consumer thread. ie. Usually the producer is so far
                # ahead of consumer that we have time to spare.
                #
                # Python is not parallel but the transform operations
                # are usually OpenCV native so release the GIL.
                #
                # Really just trying to avoid spinning up additional
                # native threads and overheads of additional
                # producer/consumer queues since this one was generally
                # idle grabbing frames.
                if self.transform:
                    frame = self.transform(frame)

                # add the frame to the queue
                self.Q.put(frame)

                self.proc_frames += 1
                if self.proc_frames >= self.frame_jump_unit:
                    self.stopped = True
                    
            else:
                time.sleep(0.1)  # Rest for 10ms, we have a full queue
        self.stream.release()
    
    
    def get_diffs(self):
        self.proc_frames = 0
        self.start()
        time.sleep(1.0)

        diffs_l1 = []
        diffs_l2 = []
        cos_sim = []
        prev_frame = None
        print(f"Group {self.group_number}: Starting process")
        while self.more():
            if self.proc_frames % 500 == 0:
                print(f"[{datetime.now()}]  Group {self.group_number}: Processed {self.proc_frames} frames")
            frame = self.read()
            if frame is None:
                print(f"Reached last frame in group {self.group_number}")
                break
            if prev_frame is not None:
                diff = frame.flatten() - prev_frame.flatten()
                diffs_l1.append(np.linalg.norm(diff, ord=1))
                diffs_l2.append(np.linalg.norm(diff))
                csim = cosine_similarity(frame.reshape(1, frame.size), prev_frame.reshape(1, prev_frame.size))[0,0]
                cos_sim.append(csim)
            prev_frame = frame

        print(f"Group {self.group_number}: Writing data")
        np.save('data/l1_' + str(self.group_number) + '.npy', np.array(diffs_l1))
        np.save('data/l2_' + str(self.group_number) + '.npy', np.array(diffs_l2))
        np.save('data/cos_' + str(self.group_number) + '.npy', np.array(cos_sim))
        print(f"Group {self.group_number}: Wrote data")

        self.stop()
        print(f"Group {self.group_number}: Finished")

        
        
def get_diffs(path, num_processes, group_number, **kwargs):
    cap = FileVideoStreamMP(path, num_processes, group_number, **kwargs)
    cap.get_diffs()
    
    

def merge_groups(path, num_groups):
    diffs_l1 = []
    diffs_l2 = []
    cos_sim = []
    cap = cv2.VideoCapture(path)
    for i in range(num_groups):
        tl1 = np.load(f'data/l1_{i}.npy')
        tl2 = np.load(f'data/l2_{i}.npy')
        tcos = np.load(f'data/cos_{i}.npy')
        diffs_l1.extend(tl1)
        diffs_l2.extend(tl2)
        cos_sim.extend(tcos)
        
        try:
            frame_jump_unit = cap.get(cv2.CAP_PROP_FRAME_COUNT) // num_groups
            frame_no = (frame_jump_unit) * (i+1) - 1
            print(frame_no)
            cap.set(1, frame_no)
            _, prev_frame = cap.read()
            _, frame = cap.read()
            diff = frame.flatten() - prev_frame.flatten()
            diffs_l1.append(np.linalg.norm(diff, ord=1))
            diffs_l2.append(np.linalg.norm(diff))
            csim = cosine_similarity(frame.reshape(1, frame.size), prev_frame.reshape(1, prev_frame.size))[0,0]
            cos_sim.append(csim)
        except Exception as e:
            print(f'Unexpected failure: {e}')
            # in the last few frames, nothing left
            pass
    cap.release()
        
    return diffs_l1, diffs_l2, cos_sim
        
    
def mask_rh_corner(frame, w, h):
    if not (isinstance(w, (float, int)) and isinstance(h, (float, int))):
        raise ValueError(f"w and h must both be float or int type, instead got w: {type(w)}, h: {type(h)}")
    if isinstance(w, float):
        w = int(frame.shape[1] * (1 - w))
    if isinstance(h, float):
        h = int(frame.shape[0] * h)

    frame[:h, w:, :] = 0
    
    return frame

if __name__ == '__main__':
    meeting_id = 220120
    path = glob.glob(f'zoom_data/{meeting_id}/*.mp4')[0]
    start = time.time()

    print(f'Starting frame diffs: {datetime.now()}\n')
    num_processes =  mp.cpu_count()
    pool = mp.Pool(num_processes)

    ### Start processes
    pool.starmap(get_diffs, [(path, num_processes, i) for i in range(num_processes)])
    print(f'Finished getting diffs.  Process took {(time.time() - start) / 60} min')

    print('Merging chunks and saving master dataset')
    start = time.time()
    diffs_l1, diffs_l2, cos_sim = merge_groups(path, num_processes)
    l1 = len(diffs_l1)
    l2 = len(diffs_l2)
    if l1 != l2:
        print(f'Result arrays are not the same length len(diffs_l1) = {l1}; len(diffs_l2) = {l2})')
        print('Truncating the larger')
        minl = min(l1, l2)
        df = pd.DataFrame({'l1': diffs_l1[:minl], 'l2': diffs_l2[:minl], 'cos_sim': cos_sim[:minl]})
    else:
        df = pd.DataFrame({'l1': diffs_l1, 'l2': diffs_l2, 'cos_sim': cos_sim})

    df['meeting_id'] = meeting_id
    df.to_csv(f'diff_data/diffs_{meeting_id}_pct_masked_cossim.csv', index=False)
    print(f'Finished merging diffs.  Process took {(time.time() - start) / 60} min')
    print(f'\nEnding time: {datetime.now()}\n')
