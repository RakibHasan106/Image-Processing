import numpy as np
import os,math,cv2        
    
def gaussian_kernel(sigma_x,kernel_size):
    """returns a gaussian blur kernel"""
    
    kernel = np.zeros((kernel_size,kernel_size))
    
    h = 1/(2.0*np.pi*sigma_x*sigma_x)
    
    n= kernel_size//2
            
    for i in range(-n,n+1):
        for j in range(-n,n+1):
            p =  ((i**2)+(j**2))/(2*(sigma_x**2))
            kernel[i+n,j+n] = h*np.exp(-p)
    
    print(kernel/np.min(kernel))
    return kernel

def convolution(img, kernel):
    n = kernel.shape[0]//2
    #img_bordered = cv2.copyMakeBorder(img, top=n , bottom=n , left=n , right=n, borderType=cv2.BORDER_CONSTANT)
    
    out = np.zeros((img.shape[0],img.shape[1],1))
    
    for x in range(n,img.shape[0]-n):
        for y in range(n, img.shape[1]-n):
            sum=0
            for i in range(-n,n+1):
                for j in range(-n,n+1):
                    sum+= img[x-i,y-j] * kernel[i+n,j+n]
            out[x,y] = sum
    
    # cv2.normalize(out,out,0,255,cv2.NORM_MINMAX)
    # out = np.round(out).astype(np.uint8)
    
    return out

def x_derivatives(sigma,kernel_size):
    kernel = np.zeros((kernel_size,kernel_size))
    
    n = kernel_size//2
    
    h = 1/(2.0*math.pi*(sigma**2))
            
    for i in range(-n,n+1):
        for j in range(-n,n+1):
            p = ((i**2)+(j**2))/(2*(sigma**2))
            kernel[i+n,j+n] = (-i/(sigma**2))*h*np.exp(-p)
    
    #print(kernel)
    return kernel

def y_derivatives(sigma,kernel_size):
    kernel = np.zeros((kernel_size,kernel_size))
    n = kernel_size//2
    
    h = 1/(2.0*math.pi*(sigma**2))
            
    for i in range(-n,n+1):
        for j in range(-n,n+1):
            p = ((i**2)+(j**2))/(2*(sigma**2))
            kernel[i+n,j+n] = (-j/(sigma**2))*h*np.exp(-p)
    
    #print(kernel)
    return kernel

def gradient_magnitude(img1,img2):
    m,n = img1.shape[0],img2.shape[1]
    output = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            output[i,j] = math.sqrt((img1[i,j]**2)+(img2[i,j]**2))
      
    return output

def non_max_suppression(img,angle):
    
    m,n=img.shape[0],img.shape[1]
    out = np.zeros(img.shape)
    img = img/img.max()*255
    
    for i in range (m-1):
        for j in range(n-1):
            try:
                
                if(0<=angle[i,j]<22.5 or 157.5<=angle[i,j]<=180):
                    q=img[i,j+1]
                    r=img[i,j-1]
                elif(22.5<=angle[i,j]<=67.5):
                    q=img[i-1,j-1]
                    r=img[i+1,j+1]
                elif(67.5<=angle[i,j]<=112.5):
                    q=img[i-1,j]
                    r=img[i+1,j]
                else:
                    q=img[i-1,j+1]
                    r=img[i+1,j-1]
                    
                if(img[i,j]<q or img[i,j]<r):
                    out[i,j]=0
                else:
                    out[i,j]=img[i,j]
                    
            except IndexError as e:
                pass
    
    return out

def find_threshold(img):
    
    oldThreshold = np.mean(img)
    
    newThreshold = threshold_generator(img,oldThreshold)
    
    while(abs(newThreshold-oldThreshold) > 0.1 ** 6):
        oldThreshold = newThreshold
        newThreshold = threshold_generator(img,oldThreshold)
        
    return newThreshold
    
    
def threshold_generator(img,threshold):
    m,n = img.shape
    
    sum1 = 0
    sum2 = 0
    n1 = 0
    n2 = 0
    
    for x in range(m):
        for y in range(n):
            if img[x,y]>threshold:
                sum1+=img[x,y]
                n1+=1
            else:
                sum2+=img[x,y]
                n2+=1
    
    highthreshold = sum1/n1
    lowthreshold = sum2/n2
    
    return (highthreshold+lowthreshold)/2
    

def doubleThresholding(img):
    
    threshold = find_threshold(img)
    
    weak = np.uint8(75)
    strong = np.uint8(255)
    
    out = np.zeros(img.shape)
    
    highThreshold = threshold * .5
    lowThreshold = highThreshold * .5
    
    strong_i,strong_j = np.where(img>=highThreshold)
    zeros_i, zeros_j = np.where(img<=lowThreshold)
    
    weak_i,weak_j = np.where((img>=lowThreshold) & (img<=highThreshold))
    
    out[strong_i,strong_j] = strong
    out[weak_i,weak_j] = weak
    out[zeros_i,zeros_j] = 0
    
    return out

def hysteresis(img):
    
    out = img.copy()
    
    weak = 75
    strong =255

    m,n = img.shape[0],img.shape[1]
    
    for i in range(1,m-1):
        for j in range(1,n-1):
            if(out[i,j]==weak):
                 out[i,j] = strong if (out[i-1,j-1]==strong or out[i-1,j]==strong or out[i-1,j+1]==strong or out[i,j-1]==strong or out[i,j+1]==strong or out[i+1,j-1]==strong or out[i+1,j]==strong or out[i+1,j+1]==strong) else 0       
    
    return out
    
def CannyEdgeDetector(img,sigma,th_high,th_low,kernel_size):
    blurred_img = convolution(img,gaussian_kernel(sigma,kernel_size))
    
    #blurred_img = cv2.filter2D(img,-1,generateGaussianKernel(sigma,sigma,7))
    
    I_x = convolution(blurred_img,x_derivatives(sigma,kernel_size)) 
    I_y = convolution(blurred_img,y_derivatives(sigma,kernel_size))
    
    I_mag = gradient_magnitude(I_x,I_y)
    
    angles = np.arctan2(I_y.copy(),I_x.copy())
    
    
    #print(angle)
    
    nms = non_max_suppression(I_mag,angles)
    
    dbl_thresholded = doubleThresholding(nms)
    
    final_output = hysteresis(dbl_thresholded)
    
    cv2.normalize(blurred_img,blurred_img,0,255,cv2.NORM_MINMAX)
    blurred_img = np.round(blurred_img).astype(np.uint8)
    
    cv2.normalize(I_x,I_x,0,255,cv2.NORM_MINMAX)
    I_x = np.round(I_x).astype(np.uint8)
    
    cv2.normalize(I_y,I_y,0,255,cv2.NORM_MINMAX)
    I_y = np.round(I_y).astype(np.uint8)
    
    cv2.normalize(I_mag,I_mag,0,255,cv2.NORM_MINMAX)
    I_mag = np.round(I_mag).astype(np.uint8)
    
    cv2.normalize(nms,nms,0,255,cv2.NORM_MINMAX)
    nms = np.round(nms).astype(np.uint8)
    
    cv2.normalize(dbl_thresholded,dbl_thresholded,0,255,cv2.NORM_MINMAX)
    dbl_thresholded = np.round(dbl_thresholded).astype(np.uint8)
    
    cv2.normalize(final_output,final_output,0,255,cv2.NORM_MINMAX)
    final_output = np.round(final_output).astype(np.uint8)
    
    
    cv2.imshow("blurred_image",blurred_img)
    cv2.imwrite("blurred_image.jpg",blurred_img)
    
    cv2.imshow("x_derivative",I_x)
    cv2.imwrite("x_derivative.jpg",I_x)
    cv2.imshow("y_derivative",I_y)
    cv2.imwrite("y_derivative.jpg",I_y)  
    
    
    cv2.imshow("magnitude",I_mag)
    cv2.imwrite("magnitude.jpg",I_mag)
    
    cv2.imshow("nms",nms)
    cv2.imwrite("nms.jpg",nms)
    
    cv2.imshow("double thresholded",dbl_thresholded)
    cv2.imwrite("double_thresholded.jpg",dbl_thresholded)
    cv2.imshow("final result after hysteresis",final_output)
    cv2.imwrite("final.jpg",final_output)
    cv2.waitKey(0)
    
    
    
    
img = cv2.imread("Lena.jpg",cv2.IMREAD_GRAYSCALE)

sigma = float(input("sigma : "))
kernel_size = int(input("kernel size : "))

CannyEdgeDetector(img,sigma,0,0,kernel_size)