import pygame
import subprocess
import time
subprocess.call(["g++", "astar.cpp"])

# Defining some colors, cell dimensions, margin, temp
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
	# outputting the map created with the help of pygame into a text file called map.txt
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
	done = False# -------- Main Program Loop -----------
	while not done:
		for event in pygame.event.get():
			if event.type == pygame.KEYDOWN:  # If user clicked a button
				done = True  # Flag that we are done so we exit this loop
			elif event.type == pygame.MOUSEBUTTONDOWN:
				# add additional rectangle area selection feature later
        	    # User clicks the mouse. Get the position
				pos = pygame.mouse.get_pos()
        	    # Change the x/y screen coordinates to grid coordinates				
				column = pos[0] // (CELL_WIDTH + MARGIN)
				row = pos[1] // (CELL_HEIGHT + MARGIN)
				# Changing location value
				if(grid[row][column]==1):
					grid[row][column] = 0
					pygame.draw.rect(screen,
            	                 RED,
                	             [(MARGIN + CELL_WIDTH) * column + MARGIN,
                    	          (MARGIN + CELL_HEIGHT) * row + MARGIN,
                        	      CELL_WIDTH,
                            	  CELL_HEIGHT])
				else:
					grid[row][column] = 1
					pygame.draw.rect(screen,
            	                 WHITE,
                	             [(MARGIN + CELL_WIDTH) * column + MARGIN,
                    	          (MARGIN + CELL_HEIGHT) * row + MARGIN,
                        	      CELL_WIDTH,
                            	  CELL_HEIGHT])
				# Limit to 60 frames per second
				clock.tick(60)
				# Go ahead and update the screen with what we've drawn.
				pygame.display.flip()


def loopfunction(grid):
	# calling / executing astar.cpp in the current python script using subprocess
	tmp=subprocess.call("./a.out")
	# A solution.txt textfile is created with solution cell coordinates present
	#taking the solution path from the updated solution.txt
	with open('solution.txt') as f:
		array = [[int(x) for x in line.split()] for line in f]
	# X and Y coordinates of solution cells
	Xarr = array[0]
	Yarr = array[1]
	if len(Xarr) > 0:
		currX = Xarr[0]
		currY = Yarr[0]
		#changing the grid cell values of the cells that are in the shortest path
		for i in range(len(Xarr)):
			grid[Xarr[i]][Yarr[i]] = 2
			color = GREEN
			pygame.draw.rect(screen, color,
                             [(MARGIN + CELL_WIDTH) * Yarr[i] + MARGIN,
                              (MARGIN + CELL_HEIGHT) * Xarr[i] + MARGIN,
                              CELL_WIDTH,
                              CELL_HEIGHT])
			# Limit to 60 frames per second
			clock.tick(60)
            # Go ahead and update the screen with what we've drawn.
			pygame.display.flip()
			currX = Xarr[i]
			currY = Yarr[i]
			exportpos(currX, currY, destX, destY)
			pause = False

			for event in pygame.event.get():   #User did something
				if event.type == pygame.MOUSEBUTTONDOWN:
					# Change the x/y screen coordinates to grid coordinates				
					pos = pygame.mouse.get_pos()
					column = pos[0] // (CELL_WIDTH + MARGIN)
					row = pos[1] // (CELL_HEIGHT + MARGIN)
					# Changing location value
					if(grid[row][column]==1):
						grid[row][column] = 0
						pygame.draw.rect(screen,
            	                 	RED,
                	             	[(MARGIN + CELL_WIDTH) * column + MARGIN,
                    	         	(MARGIN + CELL_HEIGHT) * row + MARGIN,
                        	      	CELL_WIDTH,
                            	  	CELL_HEIGHT])
					else:
						grid[row][column] = 1
						pygame.draw.rect(screen,
            	            		WHITE,
                	             	[(MARGIN + CELL_WIDTH) * column + MARGIN,
                    	          	(MARGIN + CELL_HEIGHT) * row + MARGIN,
                        	      	CELL_WIDTH,
                            	  	CELL_HEIGHT])
					# Limit to 60 frames per second
					clock.tick(60)
					# Go ahead and update the screen with what we've drawn.
					pygame.display.flip()
					pause = True
					
			if pause:
				#mapbuilder(grid)
				exportmap(grid, GRID_HEIGHT, GRID_WIDTH)
				loopfunction(grid)
				break
			time.sleep(t_step)
		exportmap(grid, GRID_HEIGHT, GRID_WIDTH)		


###################################### MAIN PROGRAM #########################################

# Creating a 2 dimensional array.
GRID_HEIGHT = int(input("ENTER HEIGHT OF THE MAP : "))
GRID_WIDTH = int(input("ENTER WIDTH OF THE MAP : "))
grid = []
for row in range(GRID_HEIGHT):
    # Add an empty array that will hold each cell in this row
    grid.append([])
    for column in range(GRID_WIDTH):
        grid[row].append(1)  # Appending a cell


# Initialize pygame
pygame.init()
# Set the CELL_HEIGHT and CELL_WIDTH of the screen
WINDOW_SIZE = [TEMP*GRID_WIDTH, TEMP*GRID_HEIGHT]
screen = pygame.display.set_mode(WINDOW_SIZE)
# Set title of screen
pygame.display.set_caption("GIVE MAP PLEASE :)")
# Used to manage how fast the screen updates
clock = pygame.time.Clock()
screen.fill(BLACK)
# Draw the grid
for row in range(GRID_HEIGHT):
	for column in range(GRID_WIDTH):
		color = WHITE
		pygame.draw.rect(screen,
            	            color,
                	        [(MARGIN + CELL_WIDTH) * column + MARGIN,
                    	     (MARGIN + CELL_HEIGHT) * row + MARGIN,
                        	  CELL_WIDTH,
                            CELL_HEIGHT])
# Limit to 60 frames per second
clock.tick(60)
# Go ahead and update the screen with what we've drawn.
pygame.display.flip()


# building map using pygame
mapbuilder(grid)
# sending this map to map.txt
exportmap(grid, GRID_HEIGHT, GRID_WIDTH)


# Taking source and destination details as input
print("Description of the Grid-input \n Top left is (0,0) and \n 1.moving right will increase j by 1 unit \n 2.moving down will increase i by 1 unit")
sourceX = int(input("Please give the source cell X coordinate : "))
sourceY = int(input("Please give the source cell Y coordinate : "))
destX = int(input("Please give the destination cell X coordinate : "))
destY = int(input("Please give the destination cell Y coordinate : "))
# sending these positions to position.txt
exportpos(sourceX, sourceY, destX, destY)


# Bot starts moving and we can change map simultaneously when it is moving
loopfunction(grid)


# Be IDLE friendly. If you forget this line, the program will 'hang' on exit.
termination = False
while not termination:
	for event in pygame.event.get():
		termination = (event.type == pygame.QUIT)
pygame.quit()

