import numpy as np
import matplotlib.pyplot as plt

def helper_plot_hist_cdf(img, ax):
    """
    Helper function to plot both Histogram and normalized CDF on the same axes.
    """
    # Calculate histogram
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    
    # Calculate CDF
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()  # Scale to match histogram height
    
    # Plot CDF (Blue line)
    ax.plot(cdf_normalized, color='b', label='CDF')
    
    # Plot Histogram (Red/Orange filled area)
    ax.hist(img.flatten(), 256, [0, 256], color='r', alpha=0.5, label='Hist')
    
    ax.set_xlim([0, 255])
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)

# ==================================================================
# QUESTION 1: Custom Histogram Equalization (Single Image Comparison)
# ==================================================================
def plot_question_1(img_orig, img_custom):
    """
    Plots the single low-contrast image against the custom HE result,
    along with their respective histograms and CDF curves.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Question 1: Custom Histogram Equalization (Single Image)", fontsize=14, fontweight='bold')
    
    # Row 1: Images
    axes[0, 0].imshow(img_orig, cmap='gray')
    axes[0, 0].set_title("Original Low-Contrast Image")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_custom, cmap='gray')
    axes[0, 1].set_title("Custom Equalized Image")
    axes[0, 1].axis('off')
    
    # Row 2: Histograms & CDFs
    axes[1, 0].set_title("Original Histogram & CDF")
    helper_plot_hist_cdf(img_orig, axes[1, 0])
    
    axes[1, 1].set_title("Custom Equalized Histogram & CDF")
    helper_plot_hist_cdf(img_custom, axes[1, 1])
    
    plt.tight_layout()
    plt.show()

# ==================================================================
# QUESTION 2: Global HE Comparison (Original vs Custom vs OpenCV)
# ==================================================================
def plot_question_2(img_orig, img_custom, img_opencv):
    """
    Plots Original, Custom HE, and OpenCV HE images side-by-side 
    with their histograms and CDF curves.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Question 2: Global HE Comparison", fontsize=14, fontweight='bold')
    
    # Row 1: Images
    axes[0, 0].imshow(img_orig, cmap='gray')
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_custom, cmap='gray')
    axes[0, 1].set_title("Custom HE Image")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img_opencv, cmap='gray')
    axes[0, 2].set_title("OpenCV HE Image")
    axes[0, 2].axis('off')
    
    # Row 2: Histograms & CDFs
    axes[1, 0].set_title("Original Hist & CDF")
    helper_plot_hist_cdf(img_orig, axes[1, 0])
    
    axes[1, 1].set_title("Custom HE Hist & CDF")
    helper_plot_hist_cdf(img_custom, axes[1, 1])
    
    axes[1, 2].set_title("OpenCV HE Hist & CDF")
    helper_plot_hist_cdf(img_opencv, axes[1, 2])
    
    plt.tight_layout()
    plt.show()

# ==================================================================
# QUESTION 4: Local vs Global Enhancement (Original vs OpenCV vs CLAHE)
# ==================================================================
def plot_question_4(img_orig, img_opencv, img_clahe):
    """
    Plots Original, Global OpenCV HE, and Local CLAHE images side-by-side
    with their histograms and CDF curves.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Question 4: Local CLAHE vs Global HE Comparison", fontsize=14, fontweight='bold')
    
    # Row 1: Images
    axes[0, 0].imshow(img_orig, cmap='gray')
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_opencv, cmap='gray')
    axes[0, 1].set_title("Global HE (OpenCV)")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img_clahe, cmap='gray')
    axes[0, 2].set_title("Local HE (CLAHE)")
    axes[0, 2].axis('off')
    # Row 2: Histograms & CDFs
    axes[1, 0].set_title("Original Hist & CDF")
    helper_plot_hist_cdf(img_orig, axes[1, 0])
    
    axes[1, 1].set_title("Global HE Hist & CDF")
    helper_plot_hist_cdf(img_opencv, axes[1, 1])
    
    axes[1, 2].set_title("CLAHE Hist & CDF")
    helper_plot_hist_cdf(img_clahe, axes[1, 2])
    
    plt.tight_layout()
    plt.show()

# ==================================================================
# QUESTION 5: Local Histogram Statistics (Gonzalez Experiments)
# ==================================================================
def plot_question_5(img_orig, 
                    res_e1, res_e2, res_e3,
                    res_k1, res_k2, res_k3,
                    res_w1, res_w2, res_w3):
    """
    Plots 9 separate experiments grouped into 3 distinct figures:
    Figure 1: Effect of Gain Constant E (C)
    Figure 2: Effect of Threshold Parameter k1
    Figure 3: Effect of Neighborhood Window Size (W)
    """
    
    # ------------------------------------------------------------------
    # FIGURE 1: Effect of Gain Constant E (C)
    # ------------------------------------------------------------------
    fig1, axes1 = plt.subplots(2, 4, figsize=(16, 8))
    fig1.suptitle("Question 5 [Part 1]: Analysis of Gain Constant E (C)", fontsize=14, fontweight='bold')
    
    axes1[0, 0].imshow(img_orig, cmap='gray'); axes1[0, 0].set_title("Original Image"); axes1[0, 0].axis('off')
    axes1[0, 1].imshow(res_e1, cmap='gray');   axes1[0, 1].set_title("Low Gain (C = 5.0)"); axes1[0, 1].axis('off')
    axes1[0, 2].imshow(res_e2, cmap='gray');   axes1[0, 2].set_title("Medium Gain (C = 15.0)"); axes1[0, 2].axis('off')
    axes1[0, 3].imshow(res_e3, cmap='gray');   axes1[0, 3].set_title("High Gain (C = 30.0)"); axes1[0, 3].axis('off')
    
    helper_plot_hist_cdf(img_orig, axes1[1, 0]); axes1[1, 0].set_title("Original Hist & CDF")
    helper_plot_hist_cdf(res_e1, axes1[1, 1]);   axes1[1, 1].set_title("C = 5.0 Hist & CDF")
    helper_plot_hist_cdf(res_e2, axes1[1, 2]);   axes1[1, 2].set_title("C = 15.0 Hist & CDF")
    helper_plot_hist_cdf(res_e3, axes1[1, 3]);   axes1[1, 3].set_title("C = 30.0 Hist & CDF")
    fig1.tight_layout()

    # ------------------------------------------------------------------
    # FIGURE 2: Effect of Threshold Parameter k
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))
    fig2.suptitle("Question 5 [Part 2]: Analysis of Std Dev Threshold (k1)", fontsize=14, fontweight='bold')
    
    axes2[0, 0].imshow(img_orig, cmap='gray'); axes2[0, 0].set_title("Original Image"); axes2[0, 0].axis('off')
    axes2[0, 1].imshow(res_k1, cmap='gray');   axes2[0, 1].set_title("Loose Lower Bound (k1 = 0.005)"); axes2[0, 1].axis('off')
    axes2[0, 2].imshow(res_k2, cmap='gray');   axes2[0, 2].set_title("Medium Bound (k1 = 0.06)"); axes2[0, 2].axis('off')
    axes2[0, 3].imshow(res_k3, cmap='gray');   axes2[0, 3].set_title("Strict Bound (k1 = 0.15)"); axes2[0, 3].axis('off')
    
    helper_plot_hist_cdf(img_orig, axes2[1, 0]); axes2[1, 0].set_title("Original Hist & CDF")
    helper_plot_hist_cdf(res_k1, axes2[1, 1]);   axes2[1, 1].set_title("k1 = 0.005 Hist & CDF")
    helper_plot_hist_cdf(res_k2, axes2[1, 2]);   axes2[1, 2].set_title("k1 = 0.06 Hist & CDF")
    helper_plot_hist_cdf(res_k3, axes2[1, 3]);   axes2[1, 3].set_title("k1 = 0.15 Hist & CDF")
    fig2.tight_layout()

    # ------------------------------------------------------------------
    # FIGURE 3: Effect of Neighborhood Window Size (W)
    # ------------------------------------------------------------------
    fig3, axes3 = plt.subplots(2, 4, figsize=(16, 8))
    fig3.suptitle("Question 5 [Part 3]: Analysis of Neighborhood Window Size (W)", fontsize=14, fontweight='bold')
    
    axes3[0, 0].imshow(img_orig, cmap='gray'); axes3[0, 0].set_title("Original Image"); axes3[0, 0].axis('off')
    axes3[0, 1].imshow(res_w1, cmap='gray');   axes3[0, 1].set_title("Small Window (3 x 3)"); axes3[0, 1].axis('off')
    axes3[0, 2].imshow(res_w2, cmap='gray');   axes3[0, 2].set_title("Medium Window (5 x 5)"); axes3[0, 2].axis('off')
    axes3[0, 3].imshow(res_w3, cmap='gray');   axes3[0, 3].set_title("Large Window (7 x 7)"); axes3[0, 3].axis('off')
    
    helper_plot_hist_cdf(img_orig, axes3[1, 0]); axes3[1, 0].set_title("Original Hist & CDF")
    helper_plot_hist_cdf(res_w1, axes3[1, 1]);   axes3[1, 1].set_title("W = 3 Hist & CDF")
    helper_plot_hist_cdf(res_w2, axes3[1, 2]);   axes3[1, 2].set_title("W = 5 Hist & CDF")
    helper_plot_hist_cdf(res_w3, axes3[1, 3]);   axes3[1, 3].set_title("W = 7 Hist & CDF")
    fig3.tight_layout()

    # this will make all 3 windows open up in same time
    plt.show()