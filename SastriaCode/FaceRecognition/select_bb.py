
import pygame, sys
from pygame.locals import K_a, K_s,K_w,K_d,K_LEFTBRACKET,K_RIGHTBRACKET

from PIL import Image
pygame.init()

BG_COLOR = (0,0,0)

def displayRect(screen, px, topleft, prior,pos,scale):
    # ensure that the rect always has positive width, height
    #topleft = [(val-pos[i])/scale for i,val in enumerate(topleft)]
    topleft = [(val/scale-pos[i]) for i,val in enumerate(topleft)]
    x, y = topleft
    bottomright = pygame.mouse.get_pos()
    width =  bottomright[0] - topleft[0]
    height = bottomright[1] - topleft[1]
    if width < 0:
        x += width
        width = abs(width)
    if height < 0:
        y += height
        height = abs(height)

    # eliminate redundant drawing cycles (when mouse isn't moving)

    current = x, y, width, height
    if not (width and height):
        return current
    if current == prior:
        return current

    # draw transparent box and blit it onto canvas
    rect = px.get_rect()
    px = pygame.transform.scale(px,[rect.width/scale, rect.height/scale])
    screen.blit(px, (rect[0]-pos[0],rect[1]-pos[1]))
    im = pygame.Surface((width, height))
    im.fill((128, 128, 128))
    pygame.draw.rect(im, (32, 32, 32), im.get_rect(), 1)
    im.set_alpha(128)
    screen.blit(im, (x, y))
    pygame.display.flip()

    # return current box extents
    return (x, y, width, height)

def setup(px):
    screen = pygame.display.set_mode( px.get_rect()[2:] )
    screen.blit(px, px.get_rect())
    pygame.display.flip()
    return screen, px

def move(pos,scale,px,screen):
    x,y = pos
    #print pos,x
    rect = px.get_rect()
    screen.fill(BG_COLOR)
    px = pygame.transform.scale(px,[rect.width/scale, rect.height/scale])
    screen.blit(px, (rect[0]-x,rect[1]-y))
    pygame.display.flip()
    #px.rect.topleft = pr.rect.topleft[0] - x, 

def mainLoop(screen, px, filelist):
    topleft = bottomright = prior = None
    n=0
    scale = 1
    pos = [0,0]
    while n!=1:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                if not topleft:
                    topleft = [(val+pos[i])*scale for i,val in enumerate(event.pos)]
                    print "tr: ",topleft
                else:
                    bottomright = [(val+pos[i])*scale for i,val in enumerate(event.pos)]
                    print "br: ",bottomright
                    n=1

            if event.type == pygame.KEYDOWN and event.key == K_a:
                pos = [pos[0]-200,pos[1]]
                move(pos,scale,px,screen)
            if event.type == pygame.KEYDOWN and event.key == K_d:
                pos = [pos[0]+200,pos[1]]
                move(pos,scale,px,screen)
            if event.type == pygame.KEYDOWN and event.key == K_w:
                pos = [pos[0],pos[1]-200]
                move(pos,scale,px,screen)
            if event.type == pygame.KEYDOWN and event.key == K_s:
                pos = [pos[0],pos[1]+200]
                move(pos,scale,px,screen)


            if event.type == pygame.KEYDOWN and event.key == K_RIGHTBRACKET:
                scale = scale/1.25
                move(pos,scale,px,screen)
            if event.type == pygame.KEYDOWN and event.key == K_LEFTBRACKET:
                scale = scale*1.25
                move(pos,scale,px,screen)


        if topleft:
            prior = displayRect(screen, px, topleft, prior,pos,scale)

    return ( topleft + bottomright )
    
if __name__ == "__main__":
    input_loc = './img/mind1176.bmp'
    output_loc = 'out.png'
    screen, px = setup(input_loc)
    left, upper, right, lower = mainLoop(screen, px)

    # ensure output rect always has positive width, height
    if right < left:
        left, right = right, left
    if lower < upper:
        lower, upper = upper, lower
    im = Image.open(input_loc)    
