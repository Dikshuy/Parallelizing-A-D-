import pygame
import subprocess
import time
subprocess.call(["g++", "astar.cpp"])

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
CELL_HEIGHT = 20
CELL_WIDTH = 20
MARGIN = 5
TEMP = 25
t_step = 1

def exportmap(grid, h, w):
	with open('map.txt', 'w') as f:
		f.write('%d' % h)
		f.write(" ")
		f.write('%d' % w)
		f.write("\n")
		for row in range(h):
			for column in range(w):
				f.write('%d' % grid[row][column])
				f.write(" ")
			f.write("\n")

def exportpos(a, b, c, d):
	with open('position.txt', 'w') as f1:
		f1.write('%d' % a)
		f1.write(" ")
		f1.write('%d' % b)
		f1.write("\n")
		f1.write('%d' % c)
		f1.write(" ")
		f1.write('%d' % d)
		f1.write("\n")	

def mapbuilder(grid):
	done = False
	while not done:
		for event in pygame.event.get():
			if event.type == pygame.KEYDOWN: 
				done = True  
			elif event.type == pygame.MOUSEBUTTONDOWN:
				pos = pygame.mouse.get_pos()				
				column = pos[0] // (CELL_WIDTH + MARGIN)
				row = pos[1] // (CELL_HEIGHT + MARGIN)
				
				if(grid[row][column]==1):
					grid[row][column] = 0
					pygame.draw.rect(screen,RED,[(MARGIN + CELL_WIDTH) * column + MARGIN,
                    	          (MARGIN + CELL_HEIGHT) * row + MARGIN,CELL_WIDTH,CELL_HEIGHT])
				else:
					grid[row][column] = 1
					pygame.draw.rect(screen,WHITE,[(MARGIN + CELL_WIDTH) * column + MARGIN,
                    	          (MARGIN + CELL_HEIGHT) * row + MARGIN,CELL_WIDTH,CELL_HEIGHT])
				
				clock.tick(60)
				pygame.display.flip()


def loopfunction(grid):
	tmp=subprocess.call("./a.out")
	with open('solution.txt') as f:
		array = [[int(x) for x in line.split()] for line in f]
	Xarr = array[0]
	Yarr = array[1]
	if len(Xarr) > 0:
		currX = Xarr[0]
		currY = Yarr[0]
		for i in range(len(Xarr)):
			grid[Xarr[i]][Yarr[i]] = 2
			color = GREEN
			pygame.draw.rect(screen, color,[(MARGIN + CELL_WIDTH) * Yarr[i] + MARGIN,
                              (MARGIN + CELL_HEIGHT) * Xarr[i] + MARGIN,CELL_WIDTH,CELL_HEIGHT])
			
			clock.tick(60)
			pygame.display.flip()
			currX = Xarr[i]
			currY = Yarr[i]
			exportpos(currX, currY, destX, destY)
			pause = False

			for event in pygame.event.get():  
				if event.type == pygame.MOUSEBUTTONDOWN:			
					pos = pygame.mouse.get_pos()
					column = pos[0] // (CELL_WIDTH + MARGIN)
					row = pos[1] // (CELL_HEIGHT + MARGIN)
					if(grid[row][column]==1):
						grid[row][column] = 0
						pygame.draw.rect(screen,RED,[(MARGIN + CELL_WIDTH) * column + MARGIN,
                    	         	(MARGIN + CELL_HEIGHT) * row + MARGIN,CELL_WIDTH,CELL_HEIGHT])
					else:
						grid[row][column] = 1
						pygame.draw.rect(screen,WHITE,[(MARGIN + CELL_WIDTH) * column + MARGIN,
                    	          	(MARGIN + CELL_HEIGHT) * row + MARGIN,CELL_WIDTH,CELL_HEIGHT])

					clock.tick(60)
					pygame.display.flip()
					pause = True
					
			if pause:
				exportmap(grid, GRID_HEIGHT, GRID_WIDTH)
				loopfunction(grid)
				break
			time.sleep(t_step)
		exportmap(grid, GRID_HEIGHT, GRID_WIDTH)		


###################################### MAIN PROGRAM #########################################

GRID_HEIGHT = int(input("ENTER HEIGHT OF THE MAP : "))
GRID_WIDTH = int(input("ENTER WIDTH OF THE MAP : "))
grid = []
for row in range(GRID_HEIGHT):
    grid.append([])
    for column in range(GRID_WIDTH):
        grid[row].append(1)  

pygame.mixer.init()
pygame.init()
WINDOW_SIZE = [TEMP*GRID_WIDTH, TEMP*GRID_HEIGHT]
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("MAP")
clock = pygame.time.Clock()

screen.fill(BLACK)
for row in range(GRID_HEIGHT):
	for column in range(GRID_WIDTH):
		pygame.draw.rect(screen,WHITE,[(MARGIN + CELL_WIDTH) * column + MARGIN,
                    	     (MARGIN + CELL_HEIGHT) * row + MARGIN,CELL_WIDTH,CELL_HEIGHT])

clock.tick(60)
pygame.display.flip()

mapbuilder(grid)
exportmap(grid, GRID_HEIGHT, GRID_WIDTH)

sourceX = int(input("Please give the source cell X coordinate : "))
sourceY = int(input("Please give the source cell Y coordinate : "))
destX = int(input("Please give the destination cell X coordinate : "))
destY = int(input("Please give the destination cell Y coordinate : "))

exportpos(sourceX, sourceY, destX, destY)

loopfunction(grid)

termination = False
while not termination:
	for event in pygame.event.get():
		termination = (event.type == pygame.QUIT)
pygame.quit()

