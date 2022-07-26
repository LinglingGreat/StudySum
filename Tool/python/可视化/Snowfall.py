from turtle import *
from random import *   
 
def ground():
    hideturtle()
    speed(100)
    for i in range(400):
        pensize(randint(5,10))
        x = randint(-400,350)
        y = randint(-280,-1)
        r = -y/280
        g = -y/280
        b = -y/280
        pencolor((r, g, b))
        penup()
        goto(x,y)
        pendown()
        forward(randint(40,100))
 
def snow():    
    hideturtle()
    pensize(2)
    speed(100)
    for i in range(100):
        r = random()
        g = random()
        b = random()
        pencolor(r, g, b)
        penup()
        setx(randint(-350,350))
        sety(randint(1,270))
        pendown()
        dens = randint(8,12)
        snowsize = randint(10,14)
        for j in range(dens):
            forward(snowsize)
            backward(snowsize)
            right(360/dens)
         
 
def main():
    setup(800, 600, 0, 0)
    tracer(False)
    bgcolor("black")    
    snow()
    ground()
    tracer(True)
    mainloop()
     
if __name__ == "__main__":
    main()