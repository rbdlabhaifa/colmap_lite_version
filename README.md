COLMAP
======

About
-----

This project is the first Structure-from-Motion (SfM) tool for ORB reconstruction on weak microprocessors (RP0)

The project-based on COLMAP, which is a general-purpose Structure-from-Motion (SfM) and Multi-View Stereo
(MVS) pipeline.

The software is licensed under the new BSD license. If you use this project for
your research, please cite:

    @inproceedings{schoenberger2016sfm,
        author={Sch\"{o}nberger, Johannes Lutz and Frahm, Jan-Michael},
        title={Structure-from-Motion Revisited},
        booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2016},
    }

    @inproceedings{schoenberger2016mvs,
        author={Sch\"{o}nberger, Johannes Lutz and Zheng, Enliang and Pollefeys, Marc and Frahm, Jan-Michael},
        title={Pixelwise View Selection for Unstructured Multi-View Stereo},
        booktitle={European Conference on Computer Vision (ECCV)},
        year={2016},
    }


The latest source code of COLMAP is available at https://github.com/colmap/colmap. 


Installation
--------

You can find details about the installation of this project in the install.rst file in this repository

Getting Started
---------------

1. First, install and build Ceres-Solver and Colmap using the instruction above.

2. Run the "camera_pose_preparation.py" script from the Orb_version folder using the command  
/camera_pose_preparation.py -p /path_to_workspace
the "path_to_workspace" folder should be empty except the .h264 video file

3. Run the ORB version bash file for the Orb_version folder using the command  
./colmap_bash.sh -p /path_to_workspace
the "path_to_workspace" folder should be empty except the .h264 video file



License
-------

The COLMAP library is licensed under the new BSD license. This text
refers only to the license for COLMAP itself, independent of its dependencies,
which are separately licensed. Building COLMAP with these dependencies may
affect the resulting COLMAP license.

    Copyright (c) 2018, ETH Zurich and UNC-Chapel Hill.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

        * Redistributions of source code must retain the above copyright
          notice, this list of conditions, and the following disclaimer.

        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.

        * Neither the name of ETH Zurich and UNC-Chapel Hill nor the names of
          its contributors may be used to endorse or promote products derived
          from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

    Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)
