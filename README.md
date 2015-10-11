# mazeSolver

### Overview ###
*   This program takes in image files then searches for and solves mazes on those images.
*   Uses opencv (python version) for the image processing and node.js for the front end.

### Image Processing Details ###

*   Uses Otsu's method and a median filter to get the initial binary mask.
*   Finds contours in the image and picks the largest contours to draw a bounding rectangle around the maze.
*   Once has the maze, uses the Zhang-Shuen algorithm for path thinning.
*   Takes the path-thinned image and finds all junctions, edges, and dead ends in the maze.
*   Finds endpoints by looking at junctions that border the edge of the maze.
*   Uses uniform cost search to find the optimal path through the maze with respect to the number of pixels.

### Node.js Details ###

* Initial screen allows you to upload an image and process it.
* After it finishes processing (~1min), will output the original and processed image with the solution overlaid on top.

### Contact ###

*   Michael Tong: mrtong96@berkeley.edu
*   Eric Zhou: eric.zhou.us@gmail.com


