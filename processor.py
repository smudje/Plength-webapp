# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 14:10:50 2017
GUI
@author: Chananchida
"""
import numpy as np
import cv2
import sknw
import networkx as nx
from sys import stdout

from PIL import Image, ImageTk

from skimage.morphology import skeletonize
from scipy import stats
from os import path

import json

class Processor:
    
    def __init__(self, master=None):
        print ("Init Processor Object")
        self.init_window()

    def test(self):
        return "Test Method Reached"
            
    def init_window(self):
        '''Creates the main window for the program'''
        # Editable
        self.c_entry = 20
        self.min_area = 500
        
        self.hue_lower, self.hue_upper = 0, 95
        self.sat_lower, self.sat_upper = 0, 255
        self.val_lower, self.val_upper = 0, 255
        
        self.side = "left"
        
        self.plant = "coleoptile"
        
    def setfile(self, filepath, filename):
        '''Opens file dialog, gets the file directory, and shows the image'''
        
                
        # More initialization
        self.filepath = filepath
        self.filename = filename
        self.img = cv2.imread(filepath)
        self.showImg(self.img)
        self.height, self.width, _ = self.img.shape
        
    def analyze(self):
        '''Does image analysis on the file and outputs the lengths'''

        self.overlay = self.img.copy()
        
        if self.plant == "coleoptile":
            img2 = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            median = cv2.medianBlur(img2, 5)
            blur = cv2.bilateralFilter(median,9,75,75)
            _,mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        elif self.plant == "seedling":
            img2 = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            median = cv2.medianBlur(img2, 5)
            blur = cv2.bilateralFilter(median,9,75,75)
            lower_thres = np.array([self.hue_lower, self.sat_lower, self.val_lower])
            upper_thres = np.array([self.hue_upper, self.sat_upper, self.val_upper])
            mask = cv2.inRange(blur, lower_thres, upper_thres)

        # Noise removal with opening and closing
        kernel = np.ones((5,5),np.uint8)
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Calibration
        if self.side == "right":
            cal = blur[self.img.shape[0]-self.img.shape[0]//5:,\
                self.img.shape[1]-self.img.shape[1]//5:]
        
        elif self.side == "left":
            cal = blur[self.img.shape[0]-self.img.shape[0]//5:,\
                :self.img.shape[1]//5]
        
        # Thresholding and stuff for calibrator
        if self.plant == "seedling":
            cal = cv2.split(cal)[2]      # Get grayscale
            
        _, calth = cv2.threshold(cal,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        calcont = cv2.findContours(calth,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]

        maxArea = 0
        for contour in calcont:
            # Calibrator is the biggest contour in that region
            if cv2.contourArea(contour) > maxArea:
                scale = cv2.boundingRect(contour)[2]
                maxArea = cv2.contourArea(contour)       


        # For displaying the results
        self.result = []
        self.firstline = "Calibration: {}mm={} pixels; 1 pixel={:.3f}mm\n\n".format(\
                      self.c_entry, scale, self.c_entry/scale)
        self.result.append(self.firstline)
        
        # Core of the program...one line
        contours = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
            
        # Progress bar
        progress = len([x for x in contours if cv2.contourArea(x) > self.min_area])
        print("Progress: [{}]".format(" "*30), end='')
        print("\b"*31, end='')
        stdout.flush()
        
        # Initialization
        cc=0
        all_results = []
        
        for contour in contours:
            # Detects only large enough contours
            if cv2.contourArea(contour) < self.min_area:
                continue
            
            cc+=1
            temp_results = [str(cc)]
            
            # Draw just one contour and skeletonize it
            temp_array = np.zeros(closing.shape, dtype=np.uint8)
            cv2.fillPoly(temp_array, [contour], color=(255,255,255))
            skeltest = skeletonize(temp_array/255)
            
            graph = sknw.build_sknw(skeltest)       # Build graph
            
            # Prune the graph only for seedlings
            if self.plant == "seedling":
                degree_list = [node for node,degree in graph.degree if degree == 1]     # List of terminal nodes
                
                                
                while degree_list:
                    node = degree_list[0]
                    neighbor = list(graph.neighbors(node))[0]
                    
                    # If the edge is shorter than a threshold and it's not the lowest node
                    # and there isn't just two nodes left...
                    if graph[node][neighbor]['weight'] < 50 and node != list(graph.nodes)[-1]\
                        and graph.degree(neighbor) != 1:
                        graph.remove_node(node)
                        
                        # If neighbor becomes terminal node, add to the list
                        if graph.degree(neighbor) == 1:
                            degree_list.append(neighbor)

                    degree_list.pop(0)
                            
                # Pruning again - remove repeat edges (RARE)
                for x, y in graph.edges:
                    if x == y:
                        graph.remove_edge(x, y)

            # Some useful values
            node, nodes = graph.nodes, list(graph.nodes)
            branch_pts = [key for key,value in graph.degree if value==3]
            branch_pts.sort(reverse=True)
            
            # TOTAL SHOOT LENGTH
            # For coleoptile, get the longest path possible
            if self.plant == "coleoptile":
                length = max(nx.single_source_dijkstra_path_length(graph,\
                        nodes[-1]).values())
                branch_pts = []
            
            # For seedling, get length from lowest to highest node (= plant length)
            elif self.plant == "seedling":
                length = nx.dijkstra_path_length(graph, nodes[-1], nodes[0])

            temp_results.append(length*self.c_entry/scale)                    
            
            # LEAF LENGTHS AND INTERNODES
            for i in range(len(branch_pts)):
                branch_pt = branch_pts[i]
                
                # Select the two "higher" nodes from the neighbors
                n1, n2, _ = sorted(graph.neighbors(branch_pt))
                
                # If it's not the last branch
                if i != len(branch_pts)-1:
            
                    # Internode length
                    interLength = nx.dijkstra_path_length(graph, branch_pt, branch_pts[i+1])
                    
                    # Determine leaf node - it's the one NOT in the longest path
                    if n1 in nx.dijkstra_path(graph, nodes[-1], nodes[0]):
                        leaf_node = self.findTerminalNode(graph, branch_pt, n2)
                        
                    else:
                        leaf_node = self.findTerminalNode(graph, branch_pt, n1)
                    
                    # Append results without correction
                    if i != 0:
                        temp_results.append(interLength*self.c_entry/scale)
                
                # For the last branch, determine leaf using distance with internode line
                else:
                    # Approximate orientation by using 100 points below the branching point
                    # Main one: if there is more than one leaf
                    if len(branch_pts) != 1:
                        # Get all the internode points
                        inter = self.mergePoints(graph, branch_pt, branch_pts[i-1])
                        
                        # If the internode is really short, use the first hundred points
                        if len(inter) < 150:
                            inter = inter[:100]
                            
                        # This one uses points a bit further down -> more accurate
                        else:
                            inter = inter[49:149]
                        
                        interApprox1 = inter[0][::-1]
                        interApprox2 = inter[-1][::-1]
                    
                    # Only one branching point, use point above the branching point
                    else:
                        path = nx.dijkstra_path(graph, nodes[-1], nodes[0])
                        
                        interApprox1 = node[branch_pt]['o'][::-1]
                        interApprox2 = graph[branch_pt][path[path.index(branch_pt)+1]]\
                                        ['pts'][-100:][0][::-1]
            
                    
                    # Compare the distance at the same y point - determine long and short node
                    l_node, l_pts, s_node, s_pts = self.compareNodes(n1, n2, graph, branch_pt)
                    
                    # Starting point for longer node
                    try:
                        start_pt = list(l_pts[:,0]).index(round(node[s_node]['o'][0]))
                                    
                    except ValueError:
                        start_pt = 0

                    # Get the maximum distance from all points on node-branch point edge
                    # to the internode line
                    s_dist = max([self.distanceWiki(interApprox1, interApprox2, each_pt[::-1])\
                                for each_pt in s_pts])        
                                
                    l_dist = max([self.distanceWiki(interApprox1, interApprox2, each_pt[::-1])\
                                for each_pt in l_pts[start_pt:]])
                    
                    # The one with the larger maximum distance is the leaf
                    if l_dist > s_dist:
                        leaf_node = l_node
                        
                    else:
                        leaf_node = s_node
                        
                leafLength = nx.dijkstra_path_length(graph, branch_pt, leaf_node)
                
                # Draw leaf
                leaf_pts = self.mergePoints(graph, leaf_node, branch_pt)
                leaf_pts = np.concatenate((leaf_pts[:,[1]], leaf_pts[:,[0]]), 1)
                cv2.polylines(self.overlay, np.int32([leaf_pts]), False, (0,255,255), 5)
        
                # For the first branching point, correct the distance using the distance
                # from the lowest node to the first branch point
                if i == 0:
                    correction = nx.dijkstra_path_length(graph, nodes[-1], branch_pt)
                    
                    # More than one branching point, there is internode
                    if len(branch_pts) != 1:
                        
                        # Fixed weighted average
                        interLength += 0.7*correction
                        temp_results.append(interLength*self.c_entry/scale)
                    
                    # Fixed weighted average
                    leafLength += 0.8*correction
                    
                temp_results.append(leafLength*self.c_entry/scale)
            
            x,y,w,h = cv2.boundingRect(contour)
    
            # Drawing bounding box and put text
            cv2.rectangle(self.overlay,(x,y),(x+w,y+h),(0,0,255),5)
            cv2.putText(self.overlay,str(cc),(x-15,y-15),cv2.FONT_HERSHEY_SIMPLEX,\
                        4, (0,0,255),5)
                        
            # Results order: total length, internode 1, leaf 1, internode 2, leaf 2, leaf 3, ...
            all_results.append(temp_results)
            
            # Progress bar :) (look at terminal)
            print('#'*(int(cc/progress*30)-int((cc-1)/progress*30)), end='')
            stdout.flush()
        
        print()
        
        self.results = {1:{'val':all_results}}          # For grouping
        self.calcStatistics()
        self.showText()
        
        # Display new image with bounding box
        b,g,r = cv2.split(self.overlay)
        temp = cv2.merge((r,g,b))
        #temp = Image.fromarray(temp)
        self.showImg(temp)
        self.flushResults()
        
        # Checkpoint
        self.analyzed = True

        return True
        
    def distanceWiki(self, P1, P2, P3):
        '''Returns the distance between point P3 and line containing points P1, P2'''
        
        return np.linalg.norm(np.cross(np.subtract(P2, P1), np.subtract(P1,P3))/\
                np.linalg.norm(np.subtract(P2,P1)))
        
    def compareNodes(self, n1, n2, graph, branch_pt):
        '''Takes two nodes and the graph as argument, returns
        (long node, points, short node, points), with points being
        the coordinates from the branch point to the terminal node'''
        
        # Get the end node and all the points in between
        t1 = self.findTerminalNode(graph, branch_pt, n1)
        p1 = self.mergePoints(graph, t1, branch_pt)

        # Do the same for n2
        t2 = self.findTerminalNode(graph, branch_pt, n2)    
        p2 = self.mergePoints(graph, t2, branch_pt)
        
        # Compare the two end nodes -> shorter node has greater y
        if graph.node[t1]['o'][0] > graph.node[t2]['o'][0]:
            return (t2, p2, t1, p1)
        
        else:
            return (t1, p1, t2, p2)
    
    def findTerminalNode(self, graph, branch_pt, node):
        '''Trace the path until the terminal node'''
        # Initialization
        all_neighbors = {branch_pt}
        x = node
        neighbors_set = set(graph.neighbors(x))
        
        # set becomes empty at the terminal node
        while neighbors_set:        
            
            all_neighbors.add(x)
            
            # Get neighbors of x and remove duplicates with main set
            neighbors_set = set(graph.neighbors(x))
            neighbors_set -= (all_neighbors&neighbors_set)
            
            # If x has new neighbors, update x as the neighbor
            if neighbors_set:
                x = sorted(list(neighbors_set))[-1]
       
        return x     
    
    def mergePoints(self, graph, n1, n2):
        '''Returns all the points from point n1 to n2'''

        # Straightforward path
        if len(nx.dijkstra_path(graph, n1, n2)) == 2:
            pts = graph[n1][n2]['pts']
        
        # Some obstacles...merge all the edges connecting them
        else:
            path = nx.dijkstra_path(graph, n1, n2)
            pts = np.concatenate(tuple(graph[path[j]][path[j+1]]['pts']\
                    for j in range(len(path)-1)))
        
        return pts
        
    
    def checkGroups(self):
        '''Removes empty groups and rearranges the groups
        so they are in chronological order'''
        
        # Remove empty groups
        keys = sorted(self.results.keys())
        
        for group_no in keys:
            if self.results[group_no]['val'] == []:
                self.results.pop(group_no)
        
        # Initialize again
        keys = sorted(self.results.keys())
        len_keys = len(keys)
        
        # If something was removed
        if len(self.results) != keys[-1]:
            for i in range(1, len_keys):
                # If the keys are not in order, rearrange them
                if keys[i] - keys[i-1] != 1:
                    self.results[i+1] = self.results.pop(keys[i])
                    keys = sorted(self.results.keys())
    
    def mergeRegions(self):
        '''Merges two or more regions together'''
        # Get text and clear entry box
        merge_text = self.e3.get()
        self.e3.delete(0, END)
        
        if not merge_text or self.grouping:
            return None
        
        merge_list = merge_text.split(',')
        
        # For each merging
        for element in merge_list:
            
            # Some initialization 
            merge_sum = 0
            pos_list=[]
            temp_merge = element.split('+')
            
            # Check within each group (no cross-group merging!)
            for group_no in self.results.keys():
                
                # For each number you have to merge
                for element2 in temp_merge:
                    
                    # Add the leaf length or total length (default = total)
                    code = 't'
                    if 'l' in element2:
                        code = element2[-2:]
                        element2 = element2[:-2]
        
                    # Find the plant to be merged
                    for pos, value in enumerate(self.results[group_no]['val']):
                        if value[0] == element2:
                            
                            # If it's the leaf, indexing is a little complicated
                            if code[0] == 'l':
                                if len(value) == (int(code[1])*2)+1:
                                    i = int(code[1])*2
                                else:
                                    i = (int(code[1])*2)+1
                            
                            # Total length is the second value of each plant
                            else:
                                i = 1
                            
                            # Add up and store positions + plant part
                            pos_list.append((pos, i))                            
                            merge_sum += value[i]
                
                # If the no. of values found is equal to no. that has to be merged
                if len(pos_list) == len(temp_merge):
                    
                    # First number in the input gets updated
                    self.results[group_no]['val'][pos_list[0][0]][0] =\
                        element
                    self.results[group_no]['val'][pos_list[0][0]][pos_list[0][1]] =\
                        merge_sum
                    
                    # The rest is removed
                    pos_list.pop(0)
                    for i in pos_list[::-1]:
                        self.results[group_no]['val'].pop(i[0])
                    
                # Reset everything
                pos_list=[]
                merge_sum = 0
                    
        # Update results
        self.calcStatistics()
        self.showText()   
        
    def selectRegions(self):
        '''Select only certain regions and removes the rest'''
        select_text = self.e3.get()
        self.e3.delete(0, END)          # Clear entry box
        
        if not select_text or self.grouping:
            return None

        # Initialization
        select_list = select_text.split(',')
        pos_list, select_list_new = [],[]

        # This is if the user enters a range, i.e. 4-9
        for element in select_list:
            temp = element.split('-')
            if len(temp) == 2:
                temp = [str(x) for x in range(int(temp[0]), int(temp[1])+1)]
    
            select_list_new += temp    
        
        # Get the positions and group numbers of selected regions
        for element in select_list_new:
            for group_no in self.results.keys():
                for pos,value in enumerate(self.results[group_no]['val']):
                    if element == value[0]:
                        pos_list.append((group_no, pos))
        
        # Get the values in those positions
        pos_list.sort()
        new_results = {i:{'val':[]} for i in self.results.keys()}
        for group_no, pos in pos_list:
            new_results[group_no]['val'].append(self.results[group_no]['val'][pos])
        
        # Update results
        self.results = new_results
        self.calcStatistics()
        self.showText()        

    
    def removeRegions(self):
        '''Removes one or more region'''
        # Get text from entry box and remove values
        remove_text = self.e3.get()
        self.e3.delete(0, END)
        
        if not remove_text or self.grouping:
            return None
        
        # Initialization
        remove_list = remove_text.split(',')
        pos_list, remove_list_new = [],[]
        
        # This is if the user enters a range, i.e. 4-9
        for element in remove_list:
            temp = element.split('-')
            if len(temp) == 2:
                temp = [str(x) for x in range(int(temp[0]), int(temp[1])+1)]

            remove_list_new += temp    
        
        # Get the positions and group numbers of regions to be removed
        for element in remove_list_new:
            for group_no in self.results.keys():
                for pos,value in enumerate(self.results[group_no]['val']):
                    if element == value[0]:
                        pos_list.append((group_no, pos))
        
        # Remove the regions from back to front
        pos_list.sort(reverse=True)
        for group_no, pos in pos_list:
            self.results[group_no]['val'].pop(pos)
        
        # Update results
        self.calcStatistics()
        self.showText()
                    
  
    def groupRegions(self):
        '''Allows the user to separate the values into groups'''
        self.grouping = True
        self.result.config(state='normal')
        
        # Replaces the calibration text with instructions
        self.result.delete(1.0,2.0)
        self.result.insert(1.0, "Please enter asterisks between groups:\n")
        
        # Get text from second line onwards and convert to list
        all_text = self.result.get(2.0, "end-1c")
        all_text = all_text.split('\n')
        
        # Remove mean, standard deviation, and standard error lines
        # Get position of the lines
        pos_list = []
        for line_no, line in enumerate(all_text):
            if line.startswith('='):
                pos_list.append(line_no)
        
        # Remove them
        pos_list.sort(reverse=True)
        for pos in range(0,len(pos_list),2):
            self.result.delete("{}.0+2l".format(pos_list[pos+1]),\
                                "{}.0+3l".format(pos_list[pos]))
        
    def applyGroups(self):
        '''Apply the groups'''
        # Get second line onwards and convert to list
        all_text = self.result.get(2.0, "end-1c")
        all_text = all_text.split('#')
        
        #  Initialization
        self.results = {}
        all_results, temp = [], []
        group_no = 1
        
        # First one will be empty from the splitting
        for element in all_text[1:]:
            
            # Keep existing groups
            if element.startswith("G"):
                # Except group 1, don't create a new group yet
                if element.split()[1] != "1":
                    self.results[group_no] = {'val':all_results}
                    group_no += 1
                    all_results = []
                    
                continue
            
            index, values = element.split(',')
            
            # Found a new group
            if values.strip().endswith('*'):
                values = values.strip().rstrip('*')
                temp = [index]
                values = [val.strip() for val in values.split(';')]
                temp += [float(length.split(':')[1].strip()) for length in values
                                                                if length != '']
                
                all_results.append(temp)
                
                # Create new group
                self.results[group_no] = {'val':all_results}
                group_no += 1
                all_results = []
                
            # Keep on collecting values until new group is found
            else:
                temp = [index]
                values = [val.strip() for val in values.split(';')]
                temp += [float(length.split(':')[1].strip()) for length in values\
                                                                if length != '']
                all_results.append(temp)
        
        # Final group
        self.results[group_no] = {'val':all_results}
        
        # Update results
        self.calcStatistics()
        self.result.delete(1.0,2.0)
        self.result.insert(1.0,self.firstline)
        self.showText()
        
        self.grouping = False


    def calcStatistics(self):
        '''Calculate mean, standard deviation, standard error for each group'''
        
        self.checkGroups()
        
        for group_no in self.results:
            
            self.results[group_no] = {'val':self.results[group_no]['val']}  # reset
            group = self.results[group_no]
            max_len = max([len(plant) for plant in group['val']])
            
            for i in range(1, max_len):
                
                # First one is always the total length
                if i == 1:
                    intresult = [plant[1] for plant in group['val']]
                    group['lengthmean'] = np.mean(intresult)
                    group['lengthsd'] = np.std(intresult, ddof=1)
                    group['lengthse'] = stats.sem(intresult)
                
                # Last one is always the leaf
                elif i == max_len-1:
                    # even and odd have different guidelines...
                    if i%2 == 0:
                        intresult = [plant[-1] for plant in group['val'] if len(plant) > i]
                        
                    elif i%2 == 1:
                        intresult = [plant[-1] for plant in group['val'] if len(plant) >= i]
                        
                    group['leaf{}mean'.format(i//2)] = np.mean(intresult)
                    group['leaf{}sd'.format(i//2)] = np.std(intresult, ddof=1)
                    group['leaf{}se'.format(i//2)] = stats.sem(intresult)
                    
                # In-between can be leaf or internode, can get kind of complicated
                else:
                    key = ['internode', 'leaf']
                    # if i is even -> internode
                    if i%2 == 0:
                        intresult = [plant[i] for plant in group['val'] if len(plant) > i+1]
                        
                    # if i is odd -> leaf
                    elif i%2 == 1:
                        intresult = [plant[i] for plant in group['val'] if len(plant) > i]
                        intresult += [plant[-1] for plant in group['val'] if len(plant) == i]
                        
                    group['{}{}mean'.format(key[i%2], i//2)] = np.mean(intresult)
                    group['{}{}sd'.format(key[i%2], i//2)] = np.std(intresult, ddof=1)
                    group['{}{}se'.format(key[i%2], i//2)] = stats.sem(intresult)
                    
        
    
    def showText(self):
        '''Displays the results in a nice fashion'''
        #self.result.config(state="normal")
        #self.result.delete(2.0, END)
        
        for group_no in sorted(self.results.keys()):
            group = self.results[group_no]
            # If there is only one group, don't print group number
            if len(self.results) != 1:
                self.result.append('\n#Group {}\n'.format(group_no))
            else:
                self.result.append('\n')
                
            
            # Print each result except the mean, SD, and stderr
            key = ['Internode', 'Leaf']
            for plant in group['val']:
                for j in range(1, len(plant)):
                    # Second value is the total length
                    if j == 1:
                        self.result.append("#{}, Length: {:.2f};\n".format(plant[0], plant[1]))
                    
                    # Final value is always the leaf length
                    elif j == len(plant)-1:
                        self.result.append("    Leaf {}: {:.2f}\n".format(j//2, plant[j]))
                    
                    # The in-between ones switch between leaf and internode
                    else:
                        self.result.append("    {} {}: {:.2f};\n".format(key[j%2],\
                        int(j/2), plant[j]))
            
            # Separator for the Mean, SD, and SE values
            self.result.append("="*30 + "\n")
            
            # Same principle as before
            # For each parameter, there exists Mean, SD, and SE -> divide by 3 to get # of parameters
            key = ['internode', 'leaf']
            for q in range(1, ((len(group)-1)//3)+1):
                
                # First one is the total length
                if q == 1:
                    self.result.append("Length: Mean, {:.2f}\n".format(group['lengthmean']))
                    self.result.append("    SD, {:.2f}\n".format(group['lengthsd']))
                    self.result.append("    SE, {:.2f}\n".format(group['lengthse']))
                
                # Last one is the leaf
                elif q == (len(group)-1)//3:
                    self.result.append("Leaf {}: Mean, {:.2f}\n".format(q//2, group['leaf{}mean'.format(q//2)]))
                    self.result.append("    SD, {:.2f}\n".format(group['leaf{}sd'.format(q//2)]))
                    self.result.append("    SE, {:.2f}\n".format(group['leaf{}se'.format(q//2)]))
                
                # In-between can be either leaf or internode
                else:
                    self.result.append("{} {}: Mean, {:.2f}\n".format(key[q%2].capitalize(), q//2,\
                            group['{}{}mean'.format(key[q%2], q//2)]))
                    self.result.append("    SD, {:.2f}\n".format(group['{}{}sd'.format(key[q%2], q//2)]))
                    self.result.append("    SE, {:.2f}\n".format(group['{}{}se'.format(key[q%2], q//2)]))
            
            self.result.append("="*30)
            
        #self.result.config(state="disabled")
        

    def showImg(self, load):
        '''Saves the image'''
        cv2.imwrite("uploads/F_" + self.filename + ".png", load)
        #fh = open("uploads/final.png", "wb")
        #fh.write(load)
    
    def flushResults(self):
        #fh = open("uploads/final.png", "w")
        #fh.write(self.results)
        with open("uploads/" + self.filename + ".json", "w") as file:
            file.write(json.dumps(self.results))
        
        
    def cropImgStart(self):
        ''' Extends the crop option'''
        if not self.filepath or self.cropping or self.analyzed or self.thresholding:
            return None
        
        self.cropping = True
        
        # Create sliders and set initial values
        self.cropleft=Scale(self, from_=0, length=275, to=self.width, orient=HORIZONTAL)
        self.cropright=Scale(self, from_=0, length=275, to=self.width, orient=HORIZONTAL)
        self.cropright.set(self.width)
        self.croptop=Scale(self, from_=0, length=275, to=self.height, orient=HORIZONTAL)
        self.cropdown=Scale(self, from_=0, length=275, to=self.height, orient=HORIZONTAL)
        self.cropdown.set(self.height)
        
        # Labels
        self.toplabel = Label(self, text='Top')
        self.downlabel = Label(self, text='Bottom')
        self.leftlabel = Label(self, text='Left')
        self.rightlabel = Label(self, text='Right')
        
        # Buttons
        self.check = Button(self, text="Check", command=self.cropImgCheck)
        self.confirm = Button(self, text="Confirm", command=self.cropImgEnd)
        
        # Layout
        self.toplabel.grid(row=2, column=2, sticky=NE, padx=10, pady=(28,0))
        self.downlabel.grid(row=2, column=2, sticky=NE, padx=10, pady=(73,0))
        self.leftlabel.grid(row=2, column=2, sticky=NE, padx=10, pady=(118,0))
        self.rightlabel.grid(row=2, column=2, sticky=NE, padx=10, pady=(163,0))
        self.croptop.grid(row=2, column=3, sticky=(W+E,NW), padx=0, pady=(10,0))
        self.cropdown.grid(row=2, column=3, sticky=(W+E,NW), padx=0, pady=(55,0))
        self.cropleft.grid(row=2, column=3, sticky=(W+E,NW), padx=0, pady=(100,0))
        self.cropright.grid(row=2, column=3, sticky=(W+E,NW), padx=0, pady=(145,0))
        self.check.grid(row=2, column=3, sticky=NW, padx=(3,0), pady=(200,0))
        self.confirm.grid(row=2, column=3, sticky=NW, padx=(60,10), pady=(200,0))
        
    def cropImgCheck(self):
        '''Checks region being cropped'''
        tempImg = self.img.copy()
        
        # Get values from the sliders
        top = self.croptop.get()
        bottom = self.cropdown.get()
        left = self.cropleft.get()
        right = self.cropright.get()
        
        # Add black rectangle over the picture
        cv2.rectangle(tempImg, (0,0), (self.width,self.height), (0, 0, 0), -1)
        
        # Add white rectangle over selected area
        cv2.rectangle(tempImg, (left,bottom), (right, top), (255, 255, 255), -1)
        
        # Overlay the rectangles with original image
        opacity = 0.5
        tempImg = cv2.addWeighted(tempImg, opacity, self.img, 1-opacity, 0)
        
        # Show the image
        b,g,r = cv2.split(tempImg)
        tempImg = cv2.merge((r,g,b))
        tempImg = Image.fromarray(tempImg)
        self.showImg(tempImg)
    
    def cropImgEnd(self):
        '''Crops the image'''
        # Get values from slider
        top = self.croptop.get()
        bottom = self.cropdown.get()
        left = self.cropleft.get()
        right = self.cropright.get()
        
        # Remove everything!
        self.toplabel.grid_remove()
        self.downlabel.grid_remove()
        self.leftlabel.grid_remove()
        self.rightlabel.grid_remove()
        self.croptop.grid_remove()
        self.cropdown.grid_remove()
        self.cropleft.grid_remove()
        self.cropright.grid_remove()
        self.check.grid_remove()
        self.confirm.grid_remove()
        
        # Actual cropping
        self.img = self.img[top:bottom, left:right]
        
        # Update information
        self.height, self.width, _ = self.img.shape
        self.imagesize.set("Image size: {} x {}".format(self.width, self.height))
        
        # Show new image
        b,g,r = cv2.split(self.img)
        temp = cv2.merge((r,g,b))
        temp = Image.fromarray(temp)
        self.showImg(temp)
        
        # Now analysis can begin!
        self.cropping = False
    
    def thresholdStart(self):
        '''Extends the threshold option'''
        
        if not self.filepath or self.cropping or self.analyzed or\
            self.thresholding or self.plant == 'coleoptile':
            return None
            
        self.thresholding = True
        
        # Create sliders and set initial values
        self.HL=Scale(self, from_=0, length=255, to=255, orient=HORIZONTAL)
        self.HR =Scale(self, from_=0, length=255, to=255, orient=HORIZONTAL)
        self.HL.set(self.hue_lower)
        self.HR.set(self.hue_upper)
        
        self.SL =Scale(self, from_=0, length=255, to=255, orient=HORIZONTAL)
        self.SR =Scale(self, from_=0, length=255, to=255, orient=HORIZONTAL)
        self.SL.set(self.sat_lower)
        self.SR.set(self.sat_upper)
        
        self.VL =Scale(self, from_=0, length=255, to=255, orient=HORIZONTAL)
        self.VR =Scale(self, from_=0, length=255, to=255, orient=HORIZONTAL)
        self.VL.set(self.val_lower)
        self.VR.set(self.val_upper)
        
        # Labels
        self.hueLabel = Label(self, text='Hue')
        self.satLabel = Label(self, text='Sat')
        self.valLabel = Label(self, text='Value')
        
        # Buttons
        self.check = Button(self, text="Check", command=self.thresholdCheck)
        self.default = Button(self, text="Reset", command=self.thresholdDefault)
        self.confirm = Button(self, text="Confirm", command=self.thresholdEnd)
        
        # Layout
        self.hueLabel.grid(row=2, column=2, sticky=NE, padx=10, pady=(28,0))
        self.satLabel.grid(row=2, column=2, sticky=NE, padx=10, pady=(118,0))
        self.valLabel.grid(row=2, column=2, sticky=NE, padx=10, pady=(208,0))
        
        self.HL.grid(row=2, column=3, sticky=(W+E,NW), padx=0, pady=(10,0))
        self.HR.grid(row=2, column=3, sticky=(W+E,NW), padx=0, pady=(55,0))
        self.SL.grid(row=2, column=3, sticky=(W+E,NW), padx=0, pady=(100,0))
        self.SR.grid(row=2, column=3, sticky=(W+E,NW), padx=0, pady=(145,0))
        self.VL.grid(row=2, column=3, sticky=(W+E,NW), padx=0, pady=(190,0))
        self.VR.grid(row=2, column=3, sticky=(W+E,NW), padx=0, pady=(235,0))
        
        self.check.grid(row=2, column=3, sticky=NW, padx=(3,0), pady=(290,0))
        self.default.grid(row=2, column=3, sticky=NW, padx=(57,10), pady=(290,0))
        self.confirm.grid(row=2, column=3, sticky=NW, padx=(107,10), pady=(290,0))
        
    def thresholdCheck(self):
        '''Checks the thresholding with given values'''
        tempImg = self.img.copy()
        tempImg2 = cv2.cvtColor(tempImg, cv2.COLOR_BGR2HSV)
        median = cv2.medianBlur(tempImg2, 5)
        blur = cv2.bilateralFilter(median,9,75,75)
        
        # Get values from the sliders
        hue_lower, hue_upper = self.HL.get(), self.HR.get()
        sat_lower, sat_upper = self.SL.get(), self.SR.get()
        val_lower, val_upper = self.VL.get(), self.VR.get()
        
        lower_thres = np.array([hue_lower, sat_lower, val_lower])
        upper_thres = np.array([hue_upper, sat_upper, val_upper])
        
        mask = cv2.inRange(blur, lower_thres, upper_thres)
        res = cv2.bitwise_and(blur, blur, mask=mask)
        
        # Show the image
        tempImg2 = cv2.cvtColor(res, cv2.COLOR_HSV2RGB)
        tempImg2 = Image.fromarray(tempImg2)
        self.showImg(tempImg2)
        
    def thresholdDefault(self):
        '''Returns the values to default'''
        
        self.HL.set(self.hue_lower)
        self.HR.set(self.hue_upper)
        self.SL.set(self.sat_lower)
        self.SR.set(self.sat_upper)
        self.VL.set(self.val_lower)
        self.VR.set(self.val_upper)
        
    
    def thresholdEnd(self):
        '''Applies the value'''
        
        # Get values from slider
        self.hue_lower, self.hue_upper = self.HL.get(), self.HR.get()
        self.sat_lower, self.sat_upper = self.SL.get(), self.SR.get()
        self.val_lower, self.val_upper = self.VL.get(), self.VR.get()
        
        # Remove everything!
        self.HL.grid_remove()
        self.HR.grid_remove()
        self.SL.grid_remove()
        self.SR.grid_remove()
        self.VL.grid_remove()
        self.VR.grid_remove()
        self.hueLabel.grid_remove()
        self.satLabel.grid_remove()
        self.valLabel.grid_remove()
        self.check.grid_remove()
        self.default.grid_remove()
        self.confirm.grid_remove()
        
        # Show original image
        b,g,r = cv2.split(self.img)
        temp = cv2.merge((r,g,b))
        temp = Image.fromarray(temp)
        self.showImg(temp)
        
        # Analysis can begin
        self.thresholding = False
    
    def exportFile(self):
        '''Exports the results in a .csv file'''
        
        # Get file path, name, and extention
        head,tail = path.split(self.filepath)
        name = tail.split('.')[0]
    
        # Write the .csv file in the same directory as the image
        file = open("{}/{}.csv".format(head,name), 'w')
        
        # Maximum no. of parameters found in all groups
        max_len = max([max([len(plant) for plant in self.results[i]['val']])\
                    for i in range(1, len(self.results)+1)])
        
        # HEADINGS
        # First column is total length
        print("Plant,Total", end='', file=file)
        
        # In between - either internode or leaf
        key = ['Internode', 'Leaf']
        for i in range(2, max_len):
            if i < max_len-1:
                print(",{} {}".format(key[i%2], i//2), end='', file=file)
            # Final column is always leaf
            else:
                print(",Leaf {}".format((i)//2), end='', file=file)
                
        print(file=file)
        
        i=2     # To prevent errors with coleoptiles
        
        for group_no in sorted(self.results.keys()):
            # Don't print group number for one group
            if len(self.results) != 1:
                print("Group {}".format(group_no), file=file)
            
            for plant in self.results[group_no]['val']:
                # Initialize plant number and total length
                temp_str = plant[0]
                temp_str += ",{:.2f}".format(plant[1])
                
            
                for i in range(2, max_len-1):
                    # For plants with lesser branching points than maximum
                    if len(plant) == i and i%2 == 1:
                        temp_str += ",{:.2f}".format(plant[-1])
                    
                    # Plant doesn't have that parameter - diff for even and odd
                    elif (len(plant) <= i+1 and i%2 == 0) or \
                        (len(plant) < i and i%2 == 1):
                        temp_str += ","
                    
                    # Normal case
                    else:
                        temp_str += ",{:.2f}".format(plant[i])
                        
                # Last one - different conditions for even and odd
                if (i%2 == 1 and len(plant) > i+1) or (i%2 == 0 and len(plant) >= i+1):
                    temp_str += ",{:.2f}".format(plant[-1])

                print(temp_str, file=file)
            
            # Number of parameters in that group
            group_len = (len(self.results[group_no])-1)//3
            
            # For the statistics
            key = ['internode', 'leaf']
            for statistics in ['Mean', 'SD', 'SE']:
                # Initialize with heading and total length
                temp_str = statistics
                temp_str += ",{:.2f}".format(self.results[group_no]\
                            ['length{}'.format(statistics.lower())])
                
                # Same idea as before
                for i in range(2, max_len-1):
                    # If the group as a whole has less params than the max param
                    if (group_len <= i and i%2==0) or (group_len+1 < i and i%2==1):
                        temp_str += ","
                    
                    # Normal case
                    else:
                        temp_str += ",{:.2f}".format(self.results[group_no]\
                            ['{}{}{}'.format(key[i%2],i//2,statistics.lower())])
                            
                # Final value
                if (group_len == i+1 and i%2==1) or (group_len >= i and i%2==0):
                    temp_str += ",{:.2f}".format(self.results[group_no]['leaf{}{}'.format\
                                ((i+1)//2, statistics.lower())])
                
                print(temp_str, file=file)
                
        self.exported.set("File has been exported as {}.csv".format(name))
                    
    def exportImg(self):
        '''Exports the labeled image'''
        
        # Get file path, name, and extension
        head, tail = path.split(self.filepath)
        name, ext = tail.split('.')
        
        # Write the labeled image in the same path as the original
        cv2.imwrite("{}/{}_labeled.{}".format(head, name, ext), self.overlay)
        
        self.exported.set("Image has been exported as {}_labeled.{}".format(\
                              name, ext))

    