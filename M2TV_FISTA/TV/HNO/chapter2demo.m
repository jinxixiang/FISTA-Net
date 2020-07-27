% This script demonstrates some basics of how to read and display
% images in MATLAB.

% Reference: See Chapter 2, 
%            "Deblurring Images - Matrices, Spectra, and Filtering"
%            by P. C. Hansen, J. G. Nagy, and D. P. O'Leary,
%            SIAM, Philadelphia, 2006.

% Switch "echo" on to display comment lines.
  echo on

% First, let's turn the "diary" on, so that we can look back later
% to see what we did.

  diary diary_chapter2
  pause

% CHOOSING AN IMAGE
  pause

% The first thing we need is an image.
% Access to the Image Processing Toolbox (IPT) provides several images we
% can use; see
%   >> help imdemos/Contents
% for a full list.  
% In addition, several images can also be downloaded from the book's website.
% For these examples, we use the images pumpkins.tif and butterflies.tif 
% from the website.

  pause

% The command "imfinfo" displays information about the image
% stored in a data file.  For example,

  info = imfinfo('butterflies.tif')

% shows that the image contained in the file butterflies.tif is
% an RGB image.  

  pause

% Doing the same thing for pumpkins.tif, we see that this image
% is a grayscale intensity image.

  infop = imfinfo('pumpkins.tif')
  pause

% READING AN IMAGE
  pause

% The command to read images in MATLAB is "imread".  
% You can find out the many ways to use this function using "help" or
% "doc"; here are two simple examples:

  G = imread('pumpkins.tif');
  F = imread('butterflies.tif');

  pause

% Now use the "whos" command to see what variables you have in your
% workspace.  

  whos
  pause

% Notice that both F and G are arrays whose entries are "uint8" values.  
% This means the intensity values are integers in the range [0,255].

% F is a three dimensional array since it contains RGB information, 
% and G is a two dimensional array since it only represents the 
% gray scale intensity values for each pixel.

  pause

% There are three basic commands for displaying images: "imshow",
% "image" and "imagesc".  
% In general, "imshow" is preferred, since it renders images more 
% accurately, especially in terms of size and color.  However, "imshow"
% can only be used if the IPT is available.   If this is not the case,
% then the commands "image" and "imagesc" can be used.
% We'll try each of these commands.

  figure(1), imshow(G), title('imshow(G)')
  pause

  figure(2), image(G), title('image(G)')
  pause

  figure(3), image(G), colormap(gray), title('image(G), colormap(gray)')
  pause

  figure(4), imagesc(G), title('imagesc(G)')
  pause

  figure(5), imagesc(G), colormap(gray), title('imagesc(G), colormap(gray)')
  pause

  figure(6), imshow(F), title('imshow(F)')
  pause

  figure(7), image(F), title('image(F)')
  pause

  figure(8), image(F), colormap(gray), title('image(F), colormap(gray)')
  pause

  figure(9), imagesc(F), title('imagesc(F)')
  pause

  figure(10), imagesc(F), colormap(gray), title('imagesc(F), colormap(gray)')
  pause


% In this example, notice that an unexpected rendering may occur.  
% This is especially true for gray scale intensity images, where 
% "image" and "imagesc" display images using a false colormap, 
% unless we explicitly specify gray using the "colormap" command.

  pause

% In addition, "image" does not always provide a proper scaling of 
% the pixel values.  Thus, if the IPT is not available, we suggest
% using the "imagesc" command.  
% In addition, because "imagesc" does not always properly set the axes, 
% we suggest following this display command with the statement "axis image".

  pause

% WRITING AN IMAGE
  pause

% To write an image to a file using any of the supported formats
% we can use the "imwrite" command.

% There are many ways to use this function, and you should
% see the on-line help for more information.  

% Here we only describe two basic approaches, which will work for 
% converting images from one data format to another; 
% for example, from TIFF to JPEG.

  pause

% This can be done simply by using "imread" to read an image
% of one format, and "imwrite" to write it to a file
% of another format.  For example:

  I = imread('pumpkins.tif');
  imwrite(I, 'pumpkins.jpg');

  pause

% Image data may also be saved in a MAT-file using the "save" command.
% In this case, if we want to use the saved image in a subsequent 
% MATLAB session, we simply use the "load" command to load the
% data into the workspace.

  save mysavefile I
  pause

% PERFORMING ARITHMETIC ON IMAGES
  pause

% We've learned the very basics of reading and writing, so now it's time 
% to learn some basics of arithmetic.  One important thing we
% must keep in mind is that most image processing software
% (this includes MATLAB) expects the pixel values to be in a fixed interval.
% Recall that typical grayscales for intensity images can have 
%   - integer values from [0, 255] or [0, 65535], 
%   - or floating point values in the interval [0, 1].
% If, after performing some arithmetic operations, the pixel values fall
% outside these intervals, unexpected results can occur.

  pause

% Moreover, since our goal is to operate on images with mathematical
% methods, integer representation of images can be limiting.  

% For example, if we multiply an image by a noninteger scalar, then the
% result contains entries which are non-integers.  

% Of course, we can easily convert these to integers by, say, rounding.  
% If we are only doing one arithmetic operation, then this approach may be
% appropriate;  the IPT provides basic image operations such as scaling.

  pause

% Let's experiment with arithmetic on images.

% First read an image:

  G = imread('pumpkins.tif');
  pause

% For the algorithms discussed later in this book, we need to understand
% how to algebraically manipulate the images; that is, we want to be able 
% to add, subtract, multiply and divide images.

% Unfortunately, standard MATLAB commands such as +, -, *, /, do not
% always work for images.  

% For example, in older versions of MATLAB (e.g., version 6.x), 
% if we attempt the simplecommand:
% 
% >>  G + 10
 
% then we get an error message.
 
%     G + 10 
%     ??? Error using ==> +
%     Function '+' is not defined for values of class 'uint8'.

  pause

% The + operator does not work for uint8 variables!  Unfortunately,
% most images stored as TIFF, JEPG, etc., are either uint8 or uint16,
% and standard arithmetic operations may not work on these types of variables.

% To get around this problem, the IPT has functions such as "imadd",
% "imsubtract", "immultiply", "imdivide" that can be used specifically
% for image operations.  

  pause

% However, we will not use these operations.

% Our algorithms require many arithmetic operations, and rounding, chopping
% and rescaling after each one of these can lead to significant
% loss of information.  

% Therefore, we adopt the convention of 
%   - converting the initial image to double-precision, 
%   - operating upon it,
%   - and then converting back to the appropriate format when
%     we are ready to display or write an image to a data file.

  pause

% In working with grayscale intensity images, the main conversion function 
% we need is "double".  It is easy to use:

  Gd = double(G);
  pause

% Use the "whos" command to see what variables are contained in the workspace.

  whos
  pause

% Notice that Gd requires significantly more memory, but now we are not 
% restricted to working only with integers, and standard arithmetic operations 
% like +, -, * and / all work in a predictable manner.

  pause

% In some cases, we may want to convert color images to grayscale
% intensity images.  This can be done by using the command "rgb2gray".
% If we then plan to use arithmetic operations on these images, then
% we will also need to convert to double.  For example:

  Fg = rgb2gray(F);
  Fd = double(Fg);
  whos
  pause

% It is not, in general, a good idea to change "true color" images to
% grayscale, since we lose information.

  pause

% In any case, once the image is converted to a double array, we can use 
% any of MATLAB's array operations on it.  For example, to determine the
% size of the image and the range of its intensities, we can use:

  size(Fd)
  pause
  max(max(Fd))
  pause
  min(min(Fd))
  pause

% DISPLAYING AND WRITING REVISITED
  pause


% Now that we can perform arithmetic on images, we need to be able to 
% display our results.  Note that Gd requires more storage space than G
% for the entries in the array although the values are really the same ;
% look at the values Gd(200,200) and G(200,200). 

  Gd(200,200)
  G(200,200)
  pause

% But try to display the image Gd using the two recommended commands,

  figure(11), imshow(Gd), title('imshow(Gd)')
  figure(12), imagesc(Gd), axis image, colormap(gray)
              title('imagesc(Gd), axis image, colormap(gray)')

% and you observe that something unusual has occurred when using "imshow".

  pause

% To understand the problem here, we need to understand how "imshow" works.
%
%  - When the input image has uint8 entries, "imshow" expects the values
%    to be integers in the range 0 (black) to 255 (white).
%
%  - When the input image has uint16 entries, it expects the values
%    to be integers in the range 0 (black) to 65535 (white).
%
%  - When the input image has double entries, it expects the values
%    to be in the range 0 (black) to 1 (white).
%
% If some entries are not in range, truncation is performed; 
%  - entries less than 0 are set to 0 (black),
%  - entries larger than the upper bound are set to the white value.
% Then the image is displayed.

  pause

% The array Gd has entries that range from 0 to 255, but they are double.
% So, before displaying the image, all entries greater than 1 are set to 1, 
% resulting in an image that has only purely black and purely white pixels.

  pause

% We can get around this in two ways.

% The first is to tell "imshow" that the max (white) and min (black) are
% different from 0 and 1:

  figure(13), imshow(Gd, [0, 255]), title('imshow(Gd, [0, 255])')
  pause

% Of course this means we need to know the max and min values in the
% array.  If we say

  figure(14), imshow(Gd, []), title('imshow(Gd, [ ])')

% then "imshow" finds the max and min values in the array,
% scales to [0,1], and then displays.

  pause

% The other way to fix this scaling problem, is to rescale Gd into
% an array with entries in [0,1], and then display:

  Gds = mat2gray(Gd);
  figure(15), imshow(Gds), title('imshow(Gds)')
  pause

% Probably the most common way we will use "imshow" is "imshow(Gd,[])",
% since it will give consistent results, even if the scaling is
% already in the interval [0,1].

  figure(16), imshow(Gd,[]), title('imshow(Gd,[ ])')
  pause


% This scaling problem must also be considered when writing images
% to one of the supported file formats using "imwrite".
% In particular, if the image array is of type double, you should
% first use the "mat2gray" command to scale the pixel values to the 
% interval [0, 1].

% For example, if Gd is an array of type double containing gray scale 
% image data, then to save the image as a JPEG or PNG file, we could use:

  imwrite(mat2gray(Gd),'MyImage.jpg','Quality',100)
  imwrite(mat2gray(Gd),'MyImage.png','BitDepth',16,'SignificantBits',16)

  pause

% If we then read the pumpkin image back with "imread", we see that
% jpeg format saves the image using only 8 bits, while png uses 16 bits.

% If 16 bits is not enough accuracy, and we want to save our images
% with their full double precision values, then we can simply use the 
% "save" command.  

% The disadvantages are that the .mat files are much larger, and they are not
% easily ported to other applications such as Java programs.

% We're finished, so we'll turn the diary off.

  diary off