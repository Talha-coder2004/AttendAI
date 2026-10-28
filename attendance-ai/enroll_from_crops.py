"""
AttendAI - enrol students from the C304_FRONT_people crops
==========================================================

1. Unzip C304_FRONT_people.zip inside the project folder, so you have
       D:\\attendence\\attendance-ai\\C304_FRONT_people\\crops\\Student_01\\...

2. Fill in names.csv (same folder as this script is run from):

       student_id,name
       Student_01,arun
       Student_05,priya
       ...

   Only list students you KNOW. Leave out anyone you can't identify
   and anyone marked "NO USABLE FACE" or "Poor" (too small to enrol).

3. Run from the project root:

       python enroll_from_crops.py

   Every face crop (main + alt) of each listed student is enlarged 3x
   and copied to  app\\face\\known_faces\\<name>\\  - the folder layout
   attendai_v6_fast.py and attendai_video_test.py already read.

Why enlarge? CCTV faces are 25-60 px. The known-face loader skips any
face smaller than 45 px, so the raw crops would be silently ignored.
Enlarging adds no detail - it only lets the detector accept them.
Real phone photos of each student (3-5 per person) are still better.
"""

import csv
import glob
import os
import sys

import cv2

CROPS_DIR = os.path.join("C304_FRONT_people", "crops")
NAMES_CSV = "names.csv"
KNOWN_FACES_DIR = os.path.join("app", "face", "known_faces")
SCALE = 3

if not os.path.isdir(CROPS_DIR):
    sys.exit(f"Not found: {CROPS_DIR}  (unzip C304_FRONT_people.zip in the project root first)")

if not os.path.isfile(NAMES_CSV):
    sys.exit(f"Not found: {NAMES_CSV}  (create it with columns student_id,name)")

added = 0

with open(NAMES_CSV, newline="", encoding="utf-8") as f:

    for row in csv.DictReader(f):

        sid = row["student_id"].strip()
        name = row["name"].strip()

        if not sid or not name:
            continue

        faces = sorted(glob.glob(os.path.join(CROPS_DIR, sid, f"{sid}_face*.jpg")))

        if not faces:
            print(f"[SKIP] {sid}: no usable face crop")
            continue

        out_dir = os.path.join(KNOWN_FACES_DIR, name)
        os.makedirs(out_dir, exist_ok=True)

        for path in faces:

            img = cv2.imread(path)

            if img is None:
                continue

            big = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)

            out = os.path.join(out_dir, f"cctv_{os.path.basename(path)}")
            cv2.imwrite(out, big)
            added += 1

        print(f"[OK] {sid} -> {out_dir}  ({len(faces)} image(s))")

print(f"\nDone. {added} image(s) added to {KNOWN_FACES_DIR}")
print("Restart attendai_v6_fast.py - it prints 'Template created: <name>' for each one it accepted.")
