import pandas as pd
import datetime
import requests
from requests.exceptions import ConnectionError
from bs4 import BeautifulSoup
import os
import matplotlib.pyplot as plt #used for plotting graphs

#Python Code made by Ishaan Vijay Puniya and Alper for Python project(r0865976 and r0847082).

def WebContentDiv(webContent,ClassPath,value):
    WebContentDiv = webContent.find_all("div",{"class":ClassPath})  # Finding the class path division.
    try:
        if value != "None":
            spans = WebContentDiv[0].find_all(value)
            texts = [span.get_text() for span in spans]
        else:
            text = WebContentDiv[0].get_text("|",strip=True)
            text = text.split("|")
            texts = text[-1]
            
    except IndexError:  #using try and except 
        texts = []
    return texts



def RealTimePrice(stockCode):  # stock code is used as parameter which I later use for enter .
    Error = 0
    url = "https://finance.yahoo.com/quote/" + stockCode + "?p=" + stockCode + "&.tsrc=fin-srch"

    try:
        r = requests.get(url)
        webContent = BeautifulSoup(r.text,"lxml")

        ##### Price and Price change
        texts = WebContentDiv(webContent,'D(ib) Mend(20px)',"fin-streamer")
        if texts != []:
            price,change = texts[0],texts[1] + " " + texts[2]
        else:
            Error = 1
            price,change = [], []

    #####VOLUME
        if stockCode[-2:] == "=F":
            texts = WebContentDiv(webContent,"D(ib) W(1/2) Bxz(bb) Pstart(12px) Va(t) ie-7_D(i) ie-7_Pos(a) smartphone_D(b) smartphone_W(100%) smartphone_Pstart(0px) smartphone_BdB smartphone_Bdc($seperatorColor)","fin-streamer")
        else:
            texts = WebContentDiv(webContent,"D(ib) W(1/2) Bxz(bb) Pend(12px) Va(t) ie-7_D(i) smartphone_D(b) smartphone_W(100%) smartphone_Pend(0px) smartphone_BdY smartphone_Bdc($seperatorColor)","fin-streamer")

        if texts != []:
            volume = texts[0]
        else:
            Error = 1
            volume = []


    #####1y target est
        texts = WebContentDiv(webContent,"D(ib) W(1/2) Bxz(bb) Pstart(12px) Va(t) ie-7_D(i) ie-7_Pos(a) smartphone_D(b) smartphone_W(100%) smartphone_Pstart(0px) smartphone_BdB smartphone_Bdc($seperatorColor)",'None') # The class of the fin streamer we used by doing inspect element.
        latestPattern,OneYearTarget = [], []



        if texts != []:
            if stockCode[-2:] == "=F":
                OneYearTarget = []
            else:
                OneYearTarget = []


        else:
            Error = 1
            OneYearTarget = []


            
    except(ConnectionError):
       price,change,volume,latestPattern,OneYearTarget = [], [], [], [], []
       Error = 1
       print("Connection Error")

    ### Provide empty list
    latestPattern = []


    return price, change, volume, latestPattern, OneYearTarget, Error #return value
#We ask for user input of Stock.
print(" 1.AAPL\n 2.ES=F\n 3.BRK-B\n 4.MU\n 5.MULN\n 6.ROO.L\n 7.TKWY.AS")
pos=input("Please enter one of the  stock codes to track price:")
pos2=input("Please enter another one of the  stock codes to track price:")
stock = [pos,pos2]
test=1
#6*2=12 
while(True and test<10):
    
    info = []
    #shift Belgium time to US time 7 hours winter, 6 hours summer
    timestamp = datetime.datetime.now() - datetime.timedelta(hours=6)
    timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    for stockCode in stock:
        stockPrice,change,volume,latestPattern,OneYearTarget,Error = RealTimePrice(stockCode)
        info.append(stockPrice)
        info.extend([change])
        info.extend([volume])
        info.extend([latestPattern])
        info.extend([OneYearTarget])
    if Error != 0:
       break


    col = [timestamp]
    col.extend(info)
    df = pd.DataFrame(col)
    df = df.T

    
    #df.to_csv(path,mode="a",header=False)
    df.to_csv(str(timestamp[0:11]) + "stock data.csv",mode="a",header = False)# 0 to 10 is the date time and this creates an csv -> excel file.
    #print(col)
    fp= pd.read_csv(str(timestamp[0:11])+"stock data.csv ", sep=',')#this is used to read the csv file
    print("           Date and Time     Price      Percentage  Volume    ")
    print("           "+pos+" "+pos2+"                                     ")#heading files.
    print(fp)
    test+=test # this is so when test reaches 10 the while loop is exited(around 5 to 6 seconds.)
    
    #The user is shown current price and change in volume of the stock entered by user.
    
#Later, we ask the user to input his buy price so they can visualize whether their investment went up or down.
ns=list(fp)
jn=float(input("Enter your  Buy Price for "+pos))
pl=[]
plt.xlabel("Stock Price")
plt.ylabel("Stock")#setting x and y labels on the graph.
pl.append(jn)
pl.append(float(ns[2])) #the index at which prices are located.
AA=['BUY PRICE','CURRENT PRICE']
plt.plot(AA,pl)
an=float(input("Enter your  Buy Price for "+pos2))
op=[]
op.append(an)
op.append(float(ns[7]))
plt.plot(AA,op)
plt.show()#function to call the graph plotted by us for this python project.
#Dear, thank you for this Amazing project.



    
