"""
Created on Wed Feb 9 22:43:08 2024

@author: Rakibul Hasan Adnan
"""
import cv2
import numpy as np
import os
import math

def LoG_kernel(sigma_x,sigma_y,kernel_size):
    """returns a LoG kernel"""  
    kernel = np.zeros((kernel_size,kernel_size))
    
    h = 1/(math.pi*(sigma_x**2)*(sigma_y**2))
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            p = -((i**2)+(j**2))/(2*sigma_x*sigma_y)
            kernel[i,j] = h*(1+p)*np.exp(p)    
            
    print(kernel)
    
    return kernel

def laplacian_kernel(kernel_size):
    """Returns the Laplacian kernel."""
    
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            if i == center and j == center:
                kernel[i, j] = -(kernel_size*kernel_size)
            else:
                kernel[i, j] = 1
    return kernel
    

def laplacian_filter(img,color,type):
    """sharpens the image"""
    
    channel = 3 if color==True else 1
        
    img_bordered = cv2.copyMakeBorder(img,top=1,bottom=1,left=1,right=1,borderType=cv2.BORDER_CONSTANT)
    
    m,n = img_bordered.shape[0]-1 , img_bordered.shape[1]-1
    
    out = np.zeros((img_bordered.shape[0],img_bordered.shape[1],channel))
    
    for x in range(1,m):
        for y in range(1,n):
            if(channel==1):
                out[x,y] =np.clip(5*img_bordered[x,y] - img_bordered[x+1,y] - img_bordered[x-1,y] - img_bordered[x,y+1] - img_bordered[x,y-1] ,0,255)
            else:
                for ch in range(channel):
                    out[x,y,ch] =np.clip(5*img_bordered[x,y,ch] - img_bordered[x+1,y,ch] - img_bordered[x-1,y,ch] - img_bordered[x,y+1,ch] - img_bordered[x,y-1,ch],0,360 if type=="hsv" else 255)
    
    cv2.normalize(out,out,0,360 if type=="hsv" else 255,cv2.NORM_MINMAX)
    out = np.round(out).astype(np.uint8)
    
    return out
   

def gaussian_kernel(sigma_x,sigma_y,kernel_size):
    """returns a gaussian blur kernel"""
    
    kernel = np.zeros((kernel_size,kernel_size))
    
    h = 1/(2.0*math.pi*sigma_x*sigma_y)
    
            
    for i in range(kernel_size):
        for j in range(kernel_size):
            p = ((i**2)/(sigma_x**2)) + ((j**2)/(sigma_y**2))
            kernel[i,j] = h*np.exp(-0.5*p)
    
    print(kernel)
    return kernel


def convolutionGray(img, kernel):
    n = kernel.shape[0]//2
    img_bordered = cv2.copyMakeBorder(img, top=n , bottom=n , left=n , right=n, borderType=cv2.BORDER_CONSTANT)
    
    out = np.zeros((img_bordered.shape[0],img_bordered.shape[1],1))
    
    for x in range(n,img_bordered.shape[0]-n):
        for y in range(n, img_bordered.shape[1]-n):
            sum=0
            for i in range(-n,n):
                for j in range(-n,n):
                    sum+= img_bordered[x-i,y-j] * kernel[i+n,j+n]
            out[x,y] = np.clip(sum,0,255)
    
    cv2.normalize(out,out,0,255,cv2.NORM_MINMAX)
    out = np.round(out).astype(np.uint8)
    
    return out
    
def convolutionColor(img,kernel,type):
    #img = cv2.imread("adnan.png",cv2.IMREAD_COLOR)
    
    ch = img.shape[2]
    n = kernel.shape[0]//2
    
    img_bordered = cv2.copyMakeBorder(img,top=n,bottom=n,left=n,right=n,borderType=cv2.BORDER_CONSTANT)
    
    out = np.ones((img_bordered.shape[0],img_bordered.shape[1],img_bordered.shape[2]))
    
    for x in range(n,img_bordered.shape[0]-n):
        for y in range(n, img_bordered.shape[1]-n):
    
            channel=0
            for channel in range(ch):
                sum=0
                for i in range(-n,n):
                    for j in range(-n,n):
                        sum+= img_bordered[x-i,y-j,channel] * kernel[i+n,j+n]
                out[x,y,channel] = np.clip(sum,0,360 if type=="hsv" else 255)
            
    
    cv2.normalize(out,out,0,360 if type=="hsv" else 255,cv2.NORM_MINMAX)
    out = np.round(out).astype(np.uint8)
    
    return out

def showDifference(img1,img2):
    """shows the difference between two image"""
    
    m,n = img1.shape[0],img2.shape[1]
    
    difference_map = np.zeros((m,n,3))
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    for x in range(m):
        for y in range(n):
            for ch in range(3):
                difference_map[x,y,ch] = abs(img1[x,y,ch]-img2[x,y,ch])
                #print(difference_map[x,y,ch])
        
    return difference_map

def sobel_kernel(size):
    """returns a horizontal sobel kernel and a vertical sobel kernel"""
    
    if size % 2 == 0:
        raise ValueError("Size must be an odd number")
    
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    
    sobel_y = np.array([
        [1, 2, 1],
        [ 0,  0,  0],
        [-1, -2, -1]
    ])
    
    return sobel_x, sobel_y


while(1):
    os.system('cls')
    
    print("What do you want to do?")
    print("1. GrayScale Image")
    print("2. Colored Image")
    print("3. RGB and HSV")
    print("Press 0 to exit")
    
    i = int(input())
    
    img = cv2.imread("Lena.jpg",cv2.IMREAD_COLOR if i == 2 or i == 3 else cv2.IMREAD_GRAYSCALE )
    if(i==3):
        img_hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        #cv2.imshow("df",img_hsv)
        output_hsv = np.zeros((img.shape[0],img.shape[1],3))
    
    if(i==1 or i==2 or i==3):
        while(1):
            os.system('cls')
            
            print("1. Laplacian Filter")
            print("2. Laplacian Filter without kernel")
            print("3. LoG Filter")
            print("4. Gaussian Blur")
            print("5. Gaussian Mean")
            print("6. Sobel Filter")
            print("7. Back")
            
            j = int(input()) 
            
            output=np.zeros((img.shape[0],img.shape[1],1 if i==1 else 3))
                
            
            if(j==7):
                break
            elif(j==6):
                m,n=sobel_kernel(5)
                    
                # print("horizontal kernel: ")
                # print(m)
                # print("vertical kernel: ")
                # print(n)
                
                # output = cv2.filter2D(img,-1,m)
                # cv2.imshow("output",output)
                # cv2.waitKey(0)
                # continue
                
                output = convolutionGray(img,n) if i==1 else convolutionColor(img,n,None)
                if(i==3):
                    output_hsv = convolutionColor(img_hsv,m,"hsv") 
                    
            elif(j==1):
                kernel_size = int(input("Kernel Size : "))
                
                kernel = laplacian_kernel(kernel_size)            
                
                output = convolutionGray(img,kernel) if i==1 else convolutionColor(img,kernel,None)
                if(i==3):
                    output_hsv = convolutionColor(img_hsv,kernel,"hsv")
                    
            elif(j==2):
                output = laplacian_filter(img,color=False,type=None) if i==1 else laplacian_filter(img,color=True,type=None)
                if(i==3):
                    output_hsv = laplacian_filter(img_hsv,color=True,type="hsv")
            
            elif(j==5):
                kernel_size = int(input("kernel size: "))
                kernel = np.ones((kernel_size,kernel_size))/(kernel_size*kernel_size)
                output= convolutionGray(img,kernel) if i==1 else convolutionColor(img,kernel,None)
                
                if(i==3):
                    output_hsv = convolutionColor(img_hsv,kernel,"hsv")
                    
            if(j==3):
                sigma_x = float(input("sigma_x = "))
                sigma_y = float(input("sigma_y = "))
                
                kernel_size = int(input("Kernel Size : "))
                
                kernel = LoG_kernel(sigma_x,sigma_y,kernel_size)
                
                output = convolutionGray(img,kernel) if i==1 else convolutionColor(img,kernel,None)
                
                if(i==3):
                    output_hsv = convolutionColor(img_hsv,kernel,"hsv")
            
            elif(j==4): 
                sigma_x = float(input("sigma_x = "))
                sigma_y = float(input("sigma_y = "))
                
                kernel_size = int(input("Kernel Size : "))
                
                kernel = gaussian_kernel(sigma_x,sigma_y,kernel_size)
                
                output = convolutionGray(img,kernel) if i==1 else convolutionColor(img,kernel,None) 
                
                if(i==3):
                    output_hsv = convolutionColor(img_hsv,kernel,"hsv")                
              
            elif(j>6 or j<1):
                continue
                   
            cv2.imshow("original image",img) if i==1 else cv2.imshow("rgb image",img)
            cv2.imwrite("input.jpg",img)
            cv2.imshow("output",output) if i==1 else cv2.imshow("output_rgb",output)
            cv2.imwrite("output.jpg",output)
            
            if(i==3):
                #img_hsv=cv2.cvtColor(img_hsv,cv2.COLOR_HSV2RGB)
                cv2.imshow("input_hsv",img_hsv)
                cv2.imwrite("input hsv.jpg",img_hsv)
                #output_hsv = cv2.cvtColor(output_hsv,cv2.COLOR_HSV2RGB)
                cv2.imshow("convoluted_hsv",output_hsv)
                cv2.imwrite("convoluted_hsv.jpg",output_hsv)
                difference = output_hsv-output
                #output_hsv = cv2.cvtColor(output_hsv,cv2.COLOR_HSV2)
                cv2.imshow("Difference",difference)
                cv2.imwrite("Difference.jpg",difference)
                
            cv2.waitKey(0)
            
    elif(i==0):
        break