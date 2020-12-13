import shutil
shutil.copy2("D:\\trial\\frames\\mygeneratedvideo.avi", "D:\\trial\\mygeneratedvideo.avi")

import os
if os.path.exists("D:\\trial\\frames\\mygeneratedvideo.avi"):
  os.remove("D:\\trial\\frames\\mygeneratedvideo.avi")