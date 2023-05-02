import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지를 인풋으로 받아서 푸리에 변환을 수행하는 함수


def ft(img, power_spectrum = True):
    img = np.array(img)
    upper_m = len(img)
    upper_n = len(img[0])

    n_n_matrix = np.exp(-2*np.pi*1j*(1/upper_n) *
                        np.array([[n*v for v in range(upper_n)] for n in range(upper_n)]))
    m_m_matrix = np.exp(-2*np.pi*1j*(1/upper_m) *
                        np.array([[m*u for m in range(upper_m)] for u in range(upper_m)]))

    img = np.matmul(img, n_n_matrix)
    img = np.matmul(m_m_matrix, img)
    if not power_spectrum:
        img = (1/(upper_m*upper_n)) * img # power_spectrum을 사용할때는 M*N을 나눔 

    return img


# 이미지를 인풋으로 받아서 inverse 푸리에 변환을 수행하는 함수
def inverse_ft(ft_img):
    img = np.array(ft_img)
    upper_m = len(img)
    upper_n = len(img[0])

    n_n_matrix = np.exp(2*np.pi*1j*(1/upper_n) *
                        np.array([[n*v for v in range(upper_n)] for n in range(upper_n)]))
    m_m_matrix = np.exp(2*np.pi*1j*(1/upper_m) *
                        np.array([[m*u for m in range(upper_m)] for u in range(upper_m)]))

    img = np.matmul(img, n_n_matrix)
    img = np.matmul(m_m_matrix, img)

    return img


# high pass 필터링 역할을 수행하는 함수
def high_pass_filtering(ft_img):
    img = np.array(ft_img)
    h = len(img)
    w = len(img[0])
    center = (h//2, w//2)
    for i in range(h):
        for j in range(w):
            distance_from_center = np.sqrt(
                (center[0]-i) * (center[0]-i) + (center[1]-j) * (center[1]-j))
            if distance_from_center < 45:
                img[i][j] = 0
    return img


# Azimuthal averaging을 수행하는 함수
def azimuthal_avg(img):
    h = len(img)
    w = len(img[0])
    center = (h//2, w//2)
    max_distance = int(
        np.ceil(np.sqrt(center[0]*center[0] + center[1]*center[1])))
    # radius 1부터 max_distance까지 key를 가진 딕셔너리 선언
    radius_dic = {radius: [] for radius in range(1, max_distance+1)}
    for i in range(h):
        for j in range(w):  # 이미지의 각 픽셀마다 센터와의 거리를 측정하여 거리에 해당하는 radius에 픽셀 값 추가
            current_radius = int(np.ceil(
                np.sqrt((center[0]-i) * (center[0]-i) + (center[1]-j) * (center[1]-j))))
            if current_radius == 0:
                current_radius = 1
            radius_dic[current_radius].append(img[i][j])

    result = []
    for r in radius_dic:
        avg_value = np.sum(radius_dic[r])/len(radius_dic[r])
        result.append(avg_value)

    max_avg = np.max(result)  # azimuthal avg의 최대값을 구하고
    result = result/max_avg  # 각 element를 최댓값으로 나누어 0~1 사이의 값으로 만든다.
    return result


# 여기서부터 main 함수
if __name__ == '__main__':

    # 이미지 이름 list
    img_list = ['000.jpg', '001.jpg', '002.jpg',
                '003.jpg', '004.jpg', '005.jpg', '006.jpg']
    one_dimension_spectrum_list =[] # 이미지들의 1D power spectrum 을 저장하기 위한 list -> 한번에 plot하기 위해


#power spectrum 을 위한 for문
    for img_name in img_list:
        img = cv2.imread('data/'+img_name, 0) # 각각의 이미지 순서대로 불러오기, flag=0 을 사용해 흑백으로 불러오기
        ft_img_for_power_spectrum = ft(img, power_spectrum = True)  # 각각의 이미지 푸리에 변환
        
        ft_img_centerized_for_power_spectrum = np.fft.fftshift(ft_img_for_power_spectrum) 
        
        ft_img_centerized_for_show = 20*np.log(np.abs(ft_img_centerized_for_power_spectrum))  # get absolute(necessary) and log (optional) for better show  

        # original 이미지 하나씩 show
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('original image'), plt.xticks([]), plt.yticks([])

        # 푸리에 변환 이미지 하나씩 show
        plt.subplot(122), plt.imshow(ft_img_centerized_for_show, cmap='gray')
        plt.title('ft_image'), plt.xticks([]), plt.yticks([])

        plt.show()
        # centerized 된 푸리에 변환 이미지를 azimuthal_avg
        one_dimension_spectrum = azimuthal_avg(ft_img_centerized_for_show) # 푸리에 변환한 이미지를 azimuthal averaging한다. 
        one_dimension_spectrum_list.append((one_dimension_spectrum, img_name)) 
    
    plt.xlim([-20, 250])
    plt.ylim([0.25, 1.2])
    for spectrum, img_name in one_dimension_spectrum_list:    
        x_axis = [x for x in range(len(spectrum))]
        plt.plot(x_axis, spectrum, label=img_name)
        
    plt.legend()
    plt.show()


#여기서부터는 역푸리에 변환을 위한 for문 
    for img_name in img_list:
        img = cv2.imread('data/'+img_name, 0) # 각각의 이미지 순서대로 불러오기, flag=0 을 사용해 흑백으로 불러오기 
        ft_img = ft(img, power_spectrum = False)# 각각의 이미지 푸리에 변환
        ft_img_centerized = np.fft.fftshift(ft_img)  # centerize 
        
        after_filter = high_pass_filtering(ft_img_centerized)  # high pass filter 적용
        
        # 필터 거친 푸리에 변환 이미지를 보여주기 위한 코드, 굳이 출력할 필요 없어서 주석 처리함. 
        '''
        after_filter_for_show = 20*np.log(np.abs(after_filter))
        '''
          
        back_img = inverse_ft(after_filter)  # inverse 푸리에 변환
        back_img_for_show = np.abs(back_img)  # get absolute for show

        # original 이미지 하나씩 show
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title(img_name), plt.xticks([]), plt.yticks([])

        # filter 거친 푸리에 변환 이미지 출력하는 코드(따로 출력할 필요는 없어보여서 주석 처리함, 출력하려면 subplot 인자 수정해야 함)
        '''
        plt.subplot(132), plt.imshow(after_filter_for_show, cmap='gray')
        plt.title(img_name), plt.xticks([]), plt.yticks([])
        '''

        # 필터 거친 후 역푸리에 변환 이미지 show
        plt.subplot(122), plt.imshow(back_img_for_show, cmap='gray')
        plt.title(img_name), plt.xticks([]), plt.yticks([])

        plt.show()