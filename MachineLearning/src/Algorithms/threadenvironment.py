# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import threading
from multiprocessing import Queue
import numpy as np

class ThreadEnvironment():
    def __init__(self, actual_nodes, node_inputs):
        self.queueLock = threading.Lock()

        self.workQueue = Queue(len(actual_nodes))
        for node_id in range(0, len(actual_nodes)):
            self.workQueue.put(node_id)
        self.exitFlag = False
        self.actual_nodes = actual_nodes
        self.node_inputs = node_inputs
        self.actual_outputs = np.zeros(len(self.actual_nodes))
        
class TaskThread(threading.Thread):
    def __init__(self, threadID, thread_environment):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.thread_environment = thread_environment
        self.print_mode = False
        
    def run(self):
        if(self.print_mode):
            print("Thread Id"+str(self.threadID)+" started: ")
        self.doTask()
    
    def doTask(self):
        while not self.thread_environment.exitFlag:
            self.thread_environment.queueLock.acquire()
            if not self.thread_environment.workQueue.empty():
                node_id = self.thread_environment.workQueue.get()
                if(self.print_mode):
                    print('Thread ', str(self.threadID), ' processing ', str(node_id))
                self.thread_environment.queueLock.release()
                self.thread_environment.actual_nodes[node_id].set_inputs(self.thread_environment.node_inputs)
            else:
                self.thread_environment.queueLock.release()
        if(self.print_mode):
            print('terminating thread ', self.threadID)
        
def process_task(nodes, node_inputs, number_of_threads):
    threads = []
    thread_environment = ThreadEnvironment(nodes, node_inputs)
    for i in range(0, number_of_threads):
        thread = TaskThread(i, thread_environment)
        thread.start()
        threads.append(thread)
        #print('Thread started...')
    
    while(not thread_environment.workQueue.empty()):
        #print(' Queue status ', thread_environment.workQueue.empty())
        pass
    
    thread_environment.exitFlag = True
    
    for t in threads:
        t.join()
    
    return thread_environment.actual_outputs

if __name__ == "__main__":
    print("Hello World")
