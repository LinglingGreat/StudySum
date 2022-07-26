from selenium import webdriver
import time
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import re
import os
import pandas as pd
import datetime


def renamefile(filepath, newfilename):
    filename = max([filepath + '\\' + f for f in os.listdir(filepath)], key=os.path.getctime)
    newfile = os.path.join(filepath, newfilename)
    if os.path.isfile(newfile):
        os.remove(newfile)
    os.rename(os.path.join(filepath, filename), newfile)


def getshfe(url, Chrome_login):
    Chrome_login.get(url)
    # yearoption_val = Chrome_login.find_element_by_class_name('ui-datepicker-year')
    # allyear = yearoption_val.find_elements_by_tag_name('option')
    # for j in range(len(allyear)):
    #     print('year: ', len(allyear), j)
    #     if int(allyear[j].text) >= 2018:
    #         continue
    # allyear = [2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004, 2003, 2002]
    allyear = [2016, 2015]
    for year in allyear:
        if year > 2018:
            continue
        else:
            # allyear[j].click()
            print(year)
            # WebDriverWait(Chrome_login, 10).until(EC.presence_of_element_located((
            #     By.XPATH, "//select[@class='ui-datepicker-year']/option[@value='" + str(year) + "']"))).click()
            Chrome_login.find_element_by_xpath("//select[@class='ui-datepicker-year']/option[@value='" + str(year) + "']").click()
            time.sleep(3)
            # itemmake_val = Chrome_login.find_element_by_class_name('ui-datepicker-month')
            # allitemmake = itemmake_val.find_elements_by_tag_name('option')
            # allmonth = [t.get_attribute("value") for t in allitemmake]
            allmonth = [i for i in range(12)]
            for month in allmonth:
                print('month: ', len(allmonth), month)
                # allitemmake[i].click()
                Chrome_login.find_element_by_xpath("//select[@class='ui-datepicker-month']/option[@value='" + str(month) + "']").click()
                hasdata = Chrome_login.find_elements_by_xpath("//td[contains(@class, ' has-data')]//a")
                allday = [t.text for t in hasdata]
                for day in allday:
                    print('day: ', day)
                    # WebDriverWait(Chrome_login, 30).until(EC.element_to_be_clickable(
                    #     (By.XPATH, "//a[text()="+str(day)+"]"))).click()
                    # WebDriverWait(Chrome_login, 30).until(EC.element_to_be_clickable(
                    #     (By.XPATH, "//a[@href='javascript:void(0)']//span[text()='全部']"))).click()
                    # WebDriverWait(Chrome_login, 30).until(EC.element_to_be_clickable(
                    #     (By.XPATH, "//a[@href='javascript:saveExcel();']"))).click()
                    Chrome_login.find_element_by_xpath("//a[text()=" + str(day) + "]").click()
                    time.sleep(3)
                    Chrome_login.find_element_by_xpath("//a[@href='javascript:void(0)']//span[text()='全部']").click()
                    time.sleep(3)
                    cElements = Chrome_login.find_element_by_xpath("//a[@href='javascript:saveExcel();']")
                    cElements.click()
                    time.sleep(3)
                    try:
                        newfilename = 'shfe' + '-' + str(year) + '-' + str(month+1) + '-' + str(day) + '.csv'
                        filepath = 'E:\\LL\\futuredata\\shfe\\'
                        renamefile(filepath, newfilename)
                    except:
                        print("Rename failed!")

                # itemmake_val = Chrome_login.find_element_by_class_name('ui-datepicker-month')
                # allitemmake = itemmake_val.find_elements_by_tag_name('option')
            # yearoption_val = Chrome_login.find_element_by_class_name('ui-datepicker-year')
            # allyear = yearoption_val.find_elements_by_tag_name('option')


def getdce(url, Chrome_login):
    Chrome_login.get(url)
    iframe = Chrome_login.find_element_by_xpath("//iframe[contains(@src,'/publicweb/quotesdata/memberDealPosiQuotes.html')]")
    Chrome_login.switch_to_frame(iframe)
    # allyear = [2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004]
    allyear = [2018, 2017]
    for year in allyear:
        if year > 2018:
            continue
        else:
            # allyear[j].click()
            # Chrome_login.find_element_by_tag_name("select")
            Chrome_login.find_element_by_xpath("//option[@value='" + str(year) + "']").click()
            # itemmake_val = Chrome_login.find_element_by_class_name('ui-datepicker-month')
            # allitemmake = Chrome_login.find_elements_by_xpath("//p[@id='control'][2]//option")
            # allmonth = [t.get_attribute("value") for t in allitemmake]
            allmonth = [i for i in range(12)]
            for month in allmonth:
                print('month: ', len(allmonth), month)
                # allitemmake[i].click()
                Chrome_login.find_element_by_xpath("//select//option[@value='" + str(month) + "']").click()
                hasdata = Chrome_login.find_elements_by_xpath("//tr[@class='week']//td")
                allday = [t.text for t in hasdata]
                while '' in allday:
                    allday.remove('')
                print(allday)
                for day in allday:
                    print('day: ', day)
                    Chrome_login.find_element_by_xpath("//tr[@class='week']//td[text()="+str(day)+"]").click()
                    time.sleep(3)
                    allpinzh = Chrome_login.find_elements_by_xpath("//li[@class='keyWord_100']//input[@type='radio']")
                    for a in range(len(allpinzh)):
                        allpinzh[a].click()
                        time.sleep(3)
                        jiaoyiri = Chrome_login.find_element_by_xpath("//div[@class='tradeResult02']//p//span").text
                        print(jiaoyiri)
                        if "非交易日" in jiaoyiri:
                            break
                        try:
                            Chrome_login.find_element_by_xpath("//a[text()='导出表格']").click()
                            time.sleep(3)
                            # newfilename = 'shfe' + '-' + str(year) + '-' + str(month + 1) + '-' + str(day) + '.csv'
                            # filepath = 'E:\\LL\\shfe'
                            # renamefile(filepath, newfilename)
                        except:
                            print("No data!")
                        allpinzh = Chrome_login.find_elements_by_xpath("//input[@type='radio']")


def getczce(url, Chrome_login, date):
    Chrome_login.get(url)
    Chrome_login.find_element_by_xpath("//input[@value='excel']").click()
    # WebDriverWait(Chrome_login, 30).until(EC.element_to_be_clickable((By.XPATH, "//input[@value='excel']"))).click()


def getczcedata(begin, end, Chrome_login):
    d = begin
    delta = datetime.timedelta(days=1)
    while d <= end:
        date = d.strftime("%Y%m%d")
        print(date)
        try:
            getczce('http://www.czce.com.cn/portal/DFSStaticFiles/Future/' + str(
                d.year) + '/' + date + '/FutureDataTradeamt.htm', Chrome_login, date)
            getczce('http://www.czce.com.cn/portal/DFSStaticFiles/Future/' + str(
                d.year) + '/' + date + '/FutureDataHolding.htm', Chrome_login, date)
        except:
            print("Today has no data!")
        d += delta


def renamexls():
    filepath = 'E:\\LL\\shfe'
    allfile = os.listdir(filepath)
    for file in allfile:
        oldfile = os.path.join(filepath, file)
        table = pd.read_excel(oldfile)
        name = table.columns[0]
        date = re.findall(r'[^()]+', name)[1]
        shotname, extension = os.path.splitext(file)
        newfile = os.path.join(filepath, date + shotname.split('(')[0] + extension)
        if os.path.isfile(newfile):
            os.remove(newfile)
        os.rename(oldfile, newfile)


if __name__ == '__main__':
    options = webdriver.ChromeOptions()
    prefs = {'download.default_directory': 'E:\\LL\\futuredata\\shfe\\'}
    options.add_experimental_option('prefs', prefs)
    Chrome_login = webdriver.Chrome(executable_path="C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe",
                                    chrome_options=options)
    # 上海期货交易所
    getshfe('http://www.shfe.com.cn/statements/dataview.html?paramid=pm', Chrome_login)
    Chrome_login.quit()

    # 大连期货交易所
    # getdce('http://www.dce.com.cn/dalianshangpin/xqsj/tjsj26/rtj/rcjccpm/index.html', Chrome_login)

    # 郑州期货交易所
    # begin = datetime.date(2016, 3, 1)
    # end = datetime.date(2016, 3, 9)
    # getczcedata(begin, end, Chrome_login)
    # time.sleep(3)
    # renamexls()



