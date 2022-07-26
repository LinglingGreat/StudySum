# ������ƺ�ʵ�ֹ���
# ���沼�����
# ��������ؼ�
#?�趨�¼���������Ӧ
# �Կؼ����в���
# ��ɳ������
from tkinter import *
import time
 
def main():
 
  # �ص�������ͨ������ָ����صĺ���
  def sendMsg():#������Ϣ
    strMsg = '��:' + time.strftime("%Y-%m-%d %H:%M:%S",
                                  time.localtime()) + '\n '
    txtMsgList.insert(END, strMsg, 'greencolor')
    txtMsgList.insert(END, txtMsg.get('0.0', END))
    txtMsg.delete('0.0', END)
     
  def cancelMsg():#ȡ����Ϣ
    txtMsg.delete('0.0', END)
 
  def sendMsgEvent(event): #������Ϣ�¼�
    if event.keysym == "Up":
      sendMsg()
 
  #�������� 
  t = Tk()
  t.title('��python������')
       
  #����frame����
  frmLT = Frame(width=500, height=320, bg='white')
  frmLC = Frame(width=500, height=150, bg='white')
  frmLB = Frame(width=500, height=30)
  frmRT = Frame(width=200, height=500)
   
  #�����ؼ�,�ؼ�������������"�ؼ�����"+"����"
  #frmLT�� frame+LeftTop
?#txtMsg�� text�ؼ�+��Ϣ
?#btnSend�� button�ؼ�+����
  txtMsgList = Text(frmLT)
  txtMsgList.tag_config('greencolor', foreground='#008C00')#����tag
  # ��Ϣ����
  txtMsg = Text(frmLC);
  txtMsg.bind("<KeyPress-Up>", sendMsgEvent)
  btnSend = Button(frmLB, text='�� ��', width = 8, command=sendMsg)
  btnCancel = Button(frmLB, text='ȡ��', width = 8, command=cancelMsg)
  # ͼ�δ���
  imgInfo = PhotoImage(file = "python.gif")
  lblImage = Label(frmRT, image = imgInfo)
  lblImage.image = imgInfo
 
  # grid()�����������Ͽؼ��Ĳ���
  #���ڲ���
  frmLT.grid(row=0, column=0, columnspan=2, padx=1, pady=3)
  frmLC.grid(row=1, column=0, columnspan=2, padx=1, pady=3)
  frmLB.grid(row=2, column=0, columnspan=2)
  frmRT.grid(row=0, column=2, rowspan=3, padx=2, pady=3)
  #�̶���С
  frmLT.grid_propagate(0)
  frmLC.grid_propagate(0)
  frmLB.grid_propagate(0)
  frmRT.grid_propagate(0)
   
  btnSend.grid(row=2, column=0)
  btnCancel.grid(row=2, column=1)
  lblImage.grid()
  txtMsgList.grid()
  txtMsg.grid()
 
  #���¼�ѭ��
  t.mainloop()
 
if __name__ == '__main__':
    main()