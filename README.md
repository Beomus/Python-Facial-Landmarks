# Facial Landmarks with Python
Simple script to detect facial landmarks using Python and OpenCV.
It can also detect faces/facial landmark in real time using the webcam.
**INPUT**
![Input](https://imgur.com/5tcH49U.jpg)
**OUTPUT**
![Output](https://imgur.com/b9bhQRb.png)

## Installation
`$ pip install opencv-python dlib imutils`

## Known Issues
- Do not work well on **ASIANS** (verified by an Asian - myself)
- Do not work well on people wearing **eye glasses, eye patches** and people with **no eyes** or **only one eye*

The _above issues_ can be fix by tweaking the `EYE_AR_THRESH` to lower for **Asians** and higher for **the rest of the world**.