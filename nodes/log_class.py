#!/usr/bin/env python
#################################################################################
#Copyright 2022 Elizabeth
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#distributed under the License is distributed on an "AS IS" BASIS,
#See the License for the specific language governing permissions and
#limitations under the License.
#################################################################################

import numpy as np
import time

class Logfile(object):
    def __init__(self,rank):
        self.rank=rank
        self.f_name=str(rank)+"_log.dat"
        self.log_file=open(self.f_name,"w")
        self.log_file.close()

    def message(self,msg):
        data_time = time.strftime("%H:%M:%S")
        out_msg = "["+str(self.rank)+"] "+data_time+" - "+ msg+"\n"
        with open(self.f_name, "a") as f:
            f.write(out_msg)
