# SurgiScan
Collaborators:
Chinmay Agrawal, cagrawal@udel.edu
Shaurya Kumar, shaurya@udel.edu
Ronith Anchan, ranchan@udel.edu, ronith.anchan@gmail.com

Presentation/Demo: https://youtu.be/R1qO1gDfMZs?si=eAENE031cV_wu5gf

Made as submission for the Patient Safety Technology Challenge at HopHacks (Johns Hopkins University)
Winner - 1st place

Surgical errors, such as accidentally leaving tools inside patients, pose significant health risks and lead to severe malpractice issues. We wanted to leverage computer vision to create a solution that enhances patient safety by reducing these preventable mistakes.

SurgiScan uses computer vision technology to detect and track surgical tools during procedures. It provides real-time monitoring to ensure that no tool is left inside a patient, improving safety and preventing malpractice.

We built SurgiScan using the YOLOv5 (You Only Look Once) object detection model, training it on a custom dataset of surgical tools. We utilized OpenCV for image processing and developed a framework to track each tool during its use, providing visual feedback to the surgical team.

Tools: Python, YOLOv5, openCV
