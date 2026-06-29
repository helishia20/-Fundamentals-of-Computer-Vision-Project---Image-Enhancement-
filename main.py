import cv2
import os
import numpy as np

from Histogram_Equalize import HistogramEqualize
from opencv_equalizeHist import opencv_eq_Hist
from Clahe import clahe
from Local_Histogram import LocalHistogramStatistics
from plotter import plot_question_5, plot_question_1, plot_question_2, plot_question_4

def run_q1(IMAGE_DIR, low_contrast_files):
    print("\n Running Question 1 (Custom H_E on first image)...")
    img1_path = os.path.join(IMAGE_DIR, low_contrast_files[0])
    img_low1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    
    if img_low1 is not None:
        img_custom1 = HistogramEqualize(img_low1)
        cv2.imwrite(os.path.join(IMAGE_DIR, "output_q1_custom.jpg"), img_custom1)
        print("-> Custom HE output for Question 1 saved successfully.")
        plot_question_1(img_low1, img_custom1)
    else:
        print(f"Error: First image named '{low_contrast_files[0]}' not found.")
    print("-" * 74)

def run_q2(IMAGE_DIR, low_contrast_files):
    print("\nRunning Question 2 (Global HE on all 3 images)...")
    for idx, filename in enumerate(low_contrast_files, start=1):
        img_path = os.path.join(IMAGE_DIR, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            continue
            
        print(f"-> Processing Question 2 for image #{idx}: {filename}")
        img_opencv = opencv_eq_Hist(img)
        cv2.imwrite(os.path.join(IMAGE_DIR, f"output_q2_opencv_img{idx}.jpg"), img_opencv)
        img_custom = HistogramEqualize(img)
        plot_question_2(img, img_custom, img_opencv)
    print("-" * 74)

def run_q4(IMAGE_DIR, low_contrast_files):
    print("\nRunning Question 4 (Local CLAHE algorithm on all 3 images)...")
    for idx, filename in enumerate(low_contrast_files, start=1):
        img_path = os.path.join(IMAGE_DIR, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            continue
            
        print(f"-> Processing Question 4 for image #{idx}: {filename}")
        img_clahe = clahe(img, clip_limit=2.0, tile_grid_size=(8, 8))
        cv2.imwrite(os.path.join(IMAGE_DIR, f"output_q4_clahe_img{idx}.jpg"), img_clahe)
        img_opencv = opencv_eq_Hist(img)
        plot_question_4(img, img_opencv, img_clahe)
    print("-" * 74)

def run_q5(IMAGE_DIR):
    print("\nRunning Question 5 (Gonzalez Local Histogram Statistics)...")
    gonzalez_path = os.path.join(IMAGE_DIR, "gonzalez_3_27.jpg")
    img_gonzalez = cv2.imread(gonzalez_path, cv2.IMREAD_GRAYSCALE)
    
    if img_gonzalez is not None:
        print("\n🔹🔹🔹Starting 9 systematic experiments. Please wait, this takes time...🔹🔹🔹")
        
        # ---part 1: chanfing C --
        print("\n🔹 [1/3] Processing C  variations...")
        print("   -> Running C = 5.0 (Window=3)...")
        res_e1 = LocalHistogramStatistics(img_gonzalez, k0=0.4, k1=0.02, k2=0.4, C=5.0,  window_size=3)
        print("   -> Running C = 15.0 (Window=3)...")
        res_e2 = LocalHistogramStatistics(img_gonzalez, k0=0.4, k1=0.02, k2=0.4, C=15.0, window_size=3)
        print("   -> Running C = 30.0 (Window=3)...")
        res_e3 = LocalHistogramStatistics(img_gonzalez, k0=0.4, k1=0.02, k2=0.4, C=30.0, window_size=3)
        
        # --- part 2: changing K ----
        print("\n🔹 [2/3] Processing k1 (Threshold) variations...")
        print("   -> Running k1 = 0.005 (Loose bound)...")
        res_k1 = LocalHistogramStatistics(img_gonzalez, k0=0.4, k1=0.005, k2=0.4, C=22.8, window_size=3)
        print("   -> Running k1 = 0.06 (Medium bound)...")
        res_k2 = LocalHistogramStatistics(img_gonzalez, k0=0.4, k1=0.06,  k2=0.4, C=22.8, window_size=3)
        print("   -> Running k1 = 0.15 (Strict bound)...")
        res_k3 = LocalHistogramStatistics(img_gonzalez, k0=0.4, k1=0.15,  k2=0.4, C=22.8, window_size=3)
        
        # --- part 3:changing w  ---
        print("\n🔹 [3/3] Processing Window Size variations (Slower)...")
        print("   -> Running Window = 3 x 3...")
        res_w1 = LocalHistogramStatistics(img_gonzalez, k0=0.4, k1=0.02, k2=0.4, C=22.8, window_size=3)
        print("   -> Running Window = 5 x 5 (Calculating, please wait)...")
        res_w2 = LocalHistogramStatistics(img_gonzalez, k0=0.4, k1=0.02, k2=0.4, C=22.8, window_size=5)
        print("   -> Running Window = 7 x 7 (Almost done, please wait)...")
        res_w3 = LocalHistogramStatistics(img_gonzalez, k0=0.4, k1=0.02, k2=0.4, C=22.8, window_size=7)
        
        print("\n🔹🔹🔹Saving all 9 generated images to disk...🔹🔹🔹")
        cv2.imwrite(os.path.join(IMAGE_DIR, "output_q5_E_05.jpg"), res_e1)
        cv2.imwrite(os.path.join(IMAGE_DIR, "output_q5_E_15.jpg"), res_e2)
        cv2.imwrite(os.path.join(IMAGE_DIR, "output_q5_E_30.jpg"), res_e3)
        
        cv2.imwrite(os.path.join(IMAGE_DIR, "output_q5_k_005.jpg"), res_k1)
        cv2.imwrite(os.path.join(IMAGE_DIR, "output_q5_k_06.jpg"),  res_k2)
        cv2.imwrite(os.path.join(IMAGE_DIR, "output_q5_k_15.jpg"),  res_k3)
        
        cv2.imwrite(os.path.join(IMAGE_DIR, "output_q5_W_3.jpg"), res_w1)
        cv2.imwrite(os.path.join(IMAGE_DIR, "output_q5_W_5.jpg"), res_w2)
        cv2.imwrite(os.path.join(IMAGE_DIR, "output_q5_W_7.jpg"), res_w3)
        print("🔹🔹🔹 Successfully saved 9 images in the 'images/' directory.🔹🔹🔹")
        
        print("\n🔹🔹🔹Rendering Matplotlib Figures...🔹🔹🔹")
        plot_question_5(img_gonzalez, 
                        res_e1, res_e2, res_e3,
                        res_k1, res_k2, res_k3,
                        res_w1, res_w2, res_w3)
    else:
        print(f"Error: Gonzalez image not found at path '{gonzalez_path}'.")
    print("-" * 74)

def main():
    IMAGE_DIR = "images"
    low_contrast_files = ["low_contrast_1.jpg", "low_contrast_2.jpg", "low_contrast_3.jpg"]
    
    while True:
        print("\n==========================================================================")
        print("         Fundamentals of Computer Vision Project - Image Enhancement       ")
        print("==========================================================================")
        print(" 1. Run Question 1 (Custom HE - Single Image)")
        print(" 2. Run Question 2 (Global HE - All 3 Images)")
        print(" 3. Run Question 4 (Local CLAHE - All 3 Images)")
        print(" 4. Run Question 5 (Gonzalez Local Histogram Statistics - 9 Exp)")
        print(" 5. Run All Questions")
        print(" 6. Exit")
        print("==========================================================================")
        
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == '1':
            run_q1(IMAGE_DIR, low_contrast_files)
        elif choice == '2':
            run_q2(IMAGE_DIR, low_contrast_files)
        elif choice == '3':
            run_q4(IMAGE_DIR, low_contrast_files)
        elif choice == '4':
            run_q5(IMAGE_DIR)
        elif choice == '5':
            run_q1(IMAGE_DIR, low_contrast_files)
            run_q2(IMAGE_DIR, low_contrast_files)
            run_q4(IMAGE_DIR, low_contrast_files)
            run_q5(IMAGE_DIR)
        elif choice == '6':
            print("\nExiting the program. Goodbye!")
            break
        else:
            print("\n❌ Invalid choice! Please enter a number between 1 and 6.")

if __name__ == '__main__':
    main()