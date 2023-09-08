
ref = 1e-3;
fac = 10;
xmax = 1.0/fac;
ymax = 0.2/fac;
ship_length = xmax/2.5;
ship_height = ymax/2.25;
xshift = xmax/2.5;
ydip = ymax/3.5;
smoth = xmax/125;
rad = xmax/125;
sug_bro = 10;

Point(1) = {0, 0, 0, ref};
Point(2) = {xmax, 0, 0, ref};
Point(3) = {xmax, ymax, 0, ref};
Point(4) = {0, ymax, 0, ref};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Point(5) = {ship_length + xshift-smoth, ymax-ydip, 0, ref/sug_bro};
Point(6) = {xshift-ship_height, ymax-ydip, 0, ref/sug_bro};
Point(7) = {xshift, ymax-ydip-ship_height, 0, ref/sug_bro};
Point(8) = {xshift, ymax-ydip, 0, ref/sug_bro};

Point(11) = {ship_length + xshift, ymax-ydip-smoth, 0, ref/sug_bro};
Point(9) = {ship_length + xshift-smoth, ymax-ydip-ship_height , 0, ref/sug_bro};
Point(10) = {ship_length + xshift, ymax-ydip-ship_height+smoth , 0, ref/sug_bro};

Point(12) = {ship_length + xshift-smoth, ymax-ydip-ship_height+smoth, 0, ref}; //bottom centre
Point(13) = {ship_length + xshift-smoth, ymax-ydip-smoth, 0, ref}; //top centre


x_1 = xmax/75;
y_1 = ymax/50; 


Point(14) = {xshift-ship_height+xmax/550, ymax-ydip+y_1, 0, ref/sug_bro};
Point(15) = {xshift-ship_height+x_1/2.5, ymax-ydip+y_1*1.25, 0, ref/sug_bro};

Point(25) = {xshift-ship_height+x_1, ymax-ydip+y_1, 0, ref/sug_bro};
Point(26) = {xshift-ship_height+2*x_1, ymax-ydip+y_1, 0, ref/sug_bro};
Point(27) = {xshift-ship_height+3*x_1, ymax-ydip+y_1, 0, ref/sug_bro};
Point(28) = {xshift-ship_height+4*x_1, ymax-ydip+y_1, 0, ref/sug_bro};
Point(29) = {xshift-ship_height+5*x_1, ymax-ydip+y_1, 0, ref/sug_bro};
Point(30) = {xshift-ship_height+6*x_1, ymax-ydip+y_1, 0, ref/sug_bro};


Line(5) = {7, 9};
Circle(6) = {9, 12, 10};
Line(7) = {10, 11};
Circle(8) = {11, 13, 5};
Line(9) = {5, 8};
Spline(11) = {6,14,15,25:30,8};
Circle(12) = {6, 8, 7};

Line Loop(1) = {1, 2, 3, 4};
Line Loop(2) = {5, 6, 7, 8, 9, -11, 12};
Plane Surface(1) = {1, 2};
// 
Physical Line(101)={1};
Physical Line(102)={2};
Physical Line(103)={3};
Physical Line(104)={4};
Physical Line(105)={5,6,7,8,9,10,11,12};
Physical Surface(201)={1};
