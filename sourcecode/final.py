import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
def data(filename):
    f= open(filename,'r')
    data=[]
    i=1
    for line in f.readlines():
        newline=line.split()
        newline[0]=i #날짜는 그냥 경과일로
        i+=1
        for index in range(1,len(newline)):
            if newline[index]=='-':
                newline[index]=0 #1000이 넘으면 1,000으로 표시돼서 정수값으로 변환
            elif len(newline[index]) >=5 and len(newline[index]) <= 7:
                a,b=map(int,newline[index].split(','))
                newline[index]=a*1000+b
            elif len(newline[index]) >=8 :
                a,b,c=map(int,newline[index].split(','))
                newline[index]=a*1000000+b*1000+c
            else:
                newline[index]=int(newline[index])
        data.append(newline)
    return data
def cof(x): #셀프피팅에 사용할 삼각함수의 계수
    return ((2*np.pi*x)/(((((x)/201)//1))*200))
def pcrline(x,liss): #학습된 곡선 함수
    return 2*liss[0]*x +8*liss[1]*(np.sin(cof(x))*((((np.abs(x)/200))//1)**2))+liss[2]
pcr=np.array(data('pcr.txt'))#pcr 날짜 전체확진자 국내발생 해외유입 사망자
vaccine=np.array(data('vaccine.txt'))#vaccine 날짜 전체1차 전체2차 전체3차 AZ1 AZ2 F1 F2 Y1 M1 M2 F3 M3 Y3
lmp='local maximum point'
#전체 확진자 추이
x=np.linspace(pcr.T[0].min(),pcr.T[0].max(),len(pcr.T[0])) #날짜 x데이터 생성
y=pcr.T[1] #확진자수 데이터 생성
plt.plot(x,y,'.')
plt.plot(x[220],y[220],'o','red',label=lmp) #지역 max point 표시
plt.plot(x[340],y[340],'o','red',label=lmp)
plt.plot(x[569],y[569],'o','red',label=lmp)
plt.title("positive scatter for whole range")
plt.xlabel('day after debut of covid19')
plt.ylabel('number of positive')
plt.legend()
plt.show()
#전체 사망자 추이
plt.title("death rate for whole range")
plt.ylabel('death rate')
plt.xlabel('day after debut of covid19')
plt.plot(x,pcr.T[4]/pcr.T[1],'-') #경과일에 대응하는 사망자/확진자 수 =>사망률
plt.show()

#위드코로나 시행 전 코로나 확진자 추이
plt.plot(x[:len(x)-50],y[:len(y)-50],'.')
plt.plot(x[220],y[220],'o','red') 
plt.plot(x[340],y[340],'o','red') #120
plt.plot(x[569],y[569],'o','red') #220
plt.xlabel('day after debut of covid and before WithCorona')
plt.ylabel('number of positive')
plt.title("positive rate before WithCorona")
plt.show()
###############################################################################################################
#확진자 라인피팅
x1=x[200:len(x)-50] #본격적으로 유행하기 시작한 날부터 위드코로나 시행 전까지
y2=y[200:len(y)-50]
poly_fit=np.polyfit(x1,y2,7) #7차 다항식으로 확진자수 학습
poly_1d=np.poly1d(poly_fit)
xs=np.linspace(x1.min(),x1.max()+50) #위드코로나 시행 후 까지 확장된 날짜변수xs
ys=poly_1d(xs)  #xs에 대응하는 확진자함수값
plt.plot(xs,np.abs(ys),'k-',label='line pitting by polyfit(opensource)') #학습된 곡선 피팅(검은색)
##############################################################################################################
###직접 확진자 곡선 짜기###
y1=np.zeros(x1.shape) ###삼각함수값을 저장할 y1열 생성
i=0
for node in x1:
    y1[i]=np.abs((np.cos(cof(node)))*((((np.abs(node)/200))//1)))#날짜 변수에 대응하는 삼각함수값 저장
    i+=1
coefs=np.vstack((x1,y1,np.ones(x1.shape))) #행렬A로 작성
coefs=np.matmul(np.linalg.pinv(coefs.T),y[:len(x1)]) #A역행렬과 실제 확진자수 열행렬을 곱하여 계수값튜플을 얻어냄
plt.plot(x1,pcrline(x1,coefs),'r-',label='line pitting by myself') #내가 직접 짠 곡선
###############################################################################################################
###위드코로나 이후 직선 만들기###
x3=x[len(x)-50:] #위드코로나 이후 데이터만 추출
y3=y[len(y)-50:] #위드코로나 이후 데이터만 추출
plt.plot(x3,y3,'y.',alpha=1,label='actual pcr positive after WithCorona') #위드코로나 이후 실제확진자 분포
ploy_fit1=np.polyfit(x3,y3,1)#위드 코로나 이후 확진자수 학습
poly_1d=np.poly1d(ploy_fit1)#위드 코로나 이후 확진자수 학습
xs1=np.linspace(x3.min(), x3.max())
ys1=poly_1d(xs1)#학습된 함수에 날짜변수를 넣은 예측확진자수
plt.plot(xs1,ys1,'y-',label='line pitting after withCorona') #위드코로나 이후 학습된 직선 그리기
###############################################################################################################
###실제 확진자 분포###

plt.plot(x[:len(x)-50],y[:len(y)-50],'b.',alpha=0.3,label='actual pcr positive')#실제 확진자 분포
###############################################################################################################
###그래프 메타데이터 값 작성
plt.plot(x[220],y[220],'o','red')
plt.plot(x[340],y[340],'o','red')
plt.plot(x[569],y[569],'o','red')
plt.annotate('local max',xy=(x[220],y[220]),arrowprops=dict(facecolor='black',shrink=0.0005,alpha=0.7))
plt.annotate('local max',xy=(x[340],y[340]),arrowprops=dict(facecolor='black',shrink=0.0005,alpha=0.7))
plt.annotate('local max',xy=(x[569],y[569]),arrowprops=dict(facecolor='black',shrink=0.0005,alpha=0.7))
plt.xlabel('day after debut of covid and before WithCorona')
plt.ylabel('number of positive')
plt.title('predicting pcr positives with line pitting')
plt.legend()
plt.show()
###############################################################################################################
# 내가 라인 피팅에 사용한 함수
plt.title('the sine wave used self line pitting')
x4=np.linspace(200,100000)
plt.plot(x4,100*np.cos(cof(x4)),'k-')
plt.legend()
plt.show()
###############################################################################################################
# 백신과 사망률 관계 추이
pop=51821669 #대한민국 총 인구 FOR 백신 접종률
###############################################################################################################
#데이터 가공
data={
      'positive':pcr[403:len(pcr)-2,1],
      'deathRate':((pcr[403:len(pcr)-2,4]/pcr[403:len(pcr)-2,1])*10e6)//1,
      'vaccine':vaccine[:,1]/pop,
      'AZ':vaccine[:,5]/pop,
      'Fizer':vaccine[:,7]/pop, #데이터에 쓰일 확진자,총 백신 접종상황, 백신별 접종 현황
      'Y':vaccine[:,8]/pop,
      'Modern':vaccine[:,10]/pop
      }
data=pd.Series(data)
###############################################################################################################
###############################################################################################################
#백신과 확진자수 분포에대한 연관성 3D 시각화
ax=plt.axes(projection='3d')
ax.set_xlabel('days')
ax.set_zlabel('number of positive')
ax.set_ylabel('vaccination of the day')
ax.view_init(10,-10) #3차원 자료 시점 변경
ax.scatter3D(np.linspace(403,len(pcr)-1,len(data['positive'])),data['vaccine'],data['positive'])#(날짜,확진자수,백신접종률)
plt.show() #해당 백신 접종률에 대응하는 확진자 수
###############################################################################################################
#경과일에 따른 백신접종률 시각화
plt.plot(np.linspace(0,len(data['vaccine'])-1,len(data['vaccine']-1)),data['vaccine'])
plt.xlabel('days')
plt.ylabel('vaccination')
plt.show()
###############################################################################################################
#백신 접종률에 따른 확진자수 2차원 시각화
plt.plot(data['vaccine'],data['positive'])
plt.xlabel('vaccination')
plt.ylabel('number of positive')
plt.show()
###############################################################################################################
#백신접종률과 사망률의 관계
x=np.array(data['vaccine']) #백신접종률과 사망률 변수 생성
y=np.array(data['deathRate']) #백신접종률과 사망률 변수 생성
plt.scatter(x,y, label='actual deathRate') #실제 백신 접종률에 따른 사망률 분포
###############################################################################################################
#백신접종률 대 사망률에 대한 라인 피팅
poly_fit=np.polyfit(x,y,4) #np의 poly_fit을 사용한 라인피팅
poly_1d=np.poly1d(poly_fit) #np의 poly_fit을 사용한 라인피팅
xs=np.linspace(x.min(),x.max()) #np의 poly_fit을 사용한 라인피팅
ys=poly_1d(xs) #np의 poly_fit을 사용한 라인피팅
plt.plot(xs,ys,color='red',label='line pitting by poly_fit')#피팅한 곡선 그리기
###############################################################################################################
#백신 접종률 대 사망률에 대한 회귀분석
formular = 'deathRate ~ vaccine' #vaccine 변수를 이용해 사망률을 학습
result=smf.ols(formular,data).fit() #statsmodels를 이용한 선형 분석
print('백신 접종률과 사망률에 대한 분석','\n',result.summary()) 
xs1=np.linspace(xs.min(),xs.max()) 
ys1=6.23e-05*xs1+5.296e+4 #학습결과에 나온 계수를 이용해 y값 입력
plt.plot(xs1,ys1,'green',label='regression by overall vaccine') #1차원에는 잘 맞지 않는다.
###############################################################################################################
#백신 별 접종률에 대한 사망률 회귀분석
formula2='deathRate~AZ+Fizer+Y+Modern' #백신별로 변수를 만들어 학습
result2=smf.ols(formula2,data).fit() #학습
print('백신별 사망률에 대한 분석','\n',result2.summary())
def deathForVaccine(A,F,Y,M): #학습 결과로 나온 계수들을 이용해 함수 작성
    return 7.365e+04 + 3.923e+04*A + 9.816e+04*F - 2.431e+06*Y + 3.382e+05*M
plt.plot(x,deathForVaccine(data['AZ'],data['Fizer'],data['Y'],data['Modern']),'k-',label='regression by sum of each vaccine')
#학습결과 곡선 플로팅
I=np.eye(4)
for i in range(4):
    now=x[-1]*I[i]
    print(deathForVaccine(now[0],now[1],now[2],now[3])) 
    #한 종류의 백신으로만 맞았을 때의 사망률
    #AZ: 1퍼센트, Fizer 1.5퍼센트, Y:음수값, Modern: 3퍼센트가 나온다
###############################################################################################################
#보정 진행
for item in ('AZ','Fizer','Modern'):
    for i in range(91,len(data[item])):
        data[item][i]=data[item][i]-data[item][i-90] #항체감소 보정, 90일 전의 접종인원을 제외한다.
fomular3='deathRate~AZ+Fizer+Modern' #AZ,화이자,모더나로만 변수를 구성,
result3=smf.ols(fomular3,data).fit() #학습
print("보정 후 결과\n",result3.summary()) #결과 출력
def deathForVaccine2(A,F,M): #학습 결과의 계수를 이용하여 함수 작성
    return 2.492e+04*A-5.058e+05*F+1.696e+06*M+7.144e+04
plt.plot(x,deathForVaccine2(data['AZ'], data['Fizer'], data['Modern']),'y-',label='result after correction')
    #보정후 학습의 결과 플로팅
I2=np.eye(3)
for i in range(3):
    vaccines=('AZ','Fizer','Modern')
    now=x[-1]*I2[i]
    print(vaccines[i],deathForVaccine2(*I2[i]))
    #보정학습 후, 한 가지 백신으로 90퍼센트의 백신접종률을 달성하였을 때 사망률 예측
    #AZ: 1퍼센트 미만, Fizer: 음수값, Modern: 2퍼센트 미만 (약 1.7)
###############################################################################################################
#그래프 메타데이터 셋
plt.legend()
plt.xlabel('vaccination rate')
plt.ylabel('death rate')
plt.title('deathRate with vaccination rate')
plt.show()

