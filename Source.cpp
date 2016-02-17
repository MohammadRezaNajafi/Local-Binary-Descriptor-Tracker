/*
Visual Object Tracking by Local Binary Descriptors
Reference paper is
[1] Minnehan, Breton, Henry Spang, and Andreas E. Savakis.
"Robust and Efficient Tracker using Dictionary of Binary Descriptors and Locality Constraints."
In Advances in Visual Computing, pp. 589-598. Springer International Publishing, 2014.

Copyright (C) 2015  Shahed Univercity (www.shahed.ac.ir)
in Tehran, Iran, Mohammad Reza Najafi.
email: <mo.najafi@shahed.ac.ir; mohammadreza.najafi1400@gmail.com>.

This file is part of LBDT (Local Binary Descriptor Tracker).

LBDT is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

LBDT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with BFROST.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
PLEASE NOTE : I AM NO LONGER ABLE TO SUPPORT THIS CODE. 
The code is quite outdated now, and relies on OpenCV 2.4.11.
LBDT Version 0.9. Licensed under LGPL, use at own risk.
*/

#include <conio.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <algorithm>


#include <opencv2\core\core.hpp>
#include <opencv2\legacy\legacy.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\nonfree\features2d.hpp>
#include <opencv2\features2d\features2d.hpp>


using namespace cv;
using namespace std;

bool		verbose = 1;				/***** This verbose variable is for showing the results during the process of tracking object *****/
float		patternScale  = 50.0;      
int			searchRadious = 30;

/*****  Use one local binary descriptor for tracking by, default is FREAK *****/
//BriefDescriptorExtractor	extractor(32);
//BRISK						extractor(30, 3, 1.5f);
FREAK						extractor(true, true, patternScale, 4);


float weightingFactor(float x, float y, float sigma2)
{
	//	float result = 	0.5 - (exp(-( (float)(x*x + y*y)/(float)(2*sigma2) ) ));
	double param = -((double)(x*x + y*y) / (double)(2 * sigma2));
	double e = exp(param);
	double result = patternScale * (0.5 - e);
	return result;
}

class FeatureTracking{
public:

	Mat static_dictionary;
	Mat dynamic_dictionary;
	Point2f object;
	Mat descriptor_object;

	int StaticDictionarySize = 15;

	BruteForceMatcher<Hamming> matcher;

	void addStaticDictionary(Mat descriptor_word);
	void addDynamicDictionary(Mat descriptor_word);

	void initialTracker(Mat img, Point2f objectInitialLocation);
	void searchObject(Mat img);

	FeatureTracking();
	~FeatureTracking();
};

void FeatureTracking::initialTracker(Mat img, Point2f objectInitialLocation)
{
	Point2f points;
	KeyPoint kp_temp;
	vector<KeyPoint> keypoints;

	vector<KeyPoint> keypoint_object;
	KeyPoint kp_object;


	int r = 30;
	int top, down, left, right;

	top = objectInitialLocation.y - r;		down = objectInitialLocation.y + r;
	left = objectInitialLocation.x - r;		right = objectInitialLocation.x + r;


	for (int i = 0; i < img.cols; i++)
	{
		for (int j = 0; j < img.rows; j++)
		{
			points.x = i;
			points.y = j;
			if ((points.x <= right && points.x >= left) && (points.y <= down && points.y >= top))
			{
				kp_temp.pt.x = i;
				kp_temp.pt.y = j;

				kp_temp.size = 1;      // 7 or 1
				kp_temp.angle = -1;

				keypoints.push_back(kp_temp);
//				circle(img, points, 2, CV_RGB(250, 0, 0), 2, 8, 0);

			}
		}
	}

//	BriefDescriptorExtractor extractor(64);
//	BRISK extractor(30, 3, 1.5f);
//	FREAK extractor(true, true, patternScale, 4);
	Mat descriptors;
	extractor.compute(img, keypoints, descriptors);

	kp_object.pt.x = objectInitialLocation.x;
	kp_object.pt.y = objectInitialLocation.y;
	kp_object.size = 1;
	kp_object.angle = -1;

	keypoint_object.push_back(kp_object);
	Mat descriptor_objectInMethod;
	extractor.compute(img, keypoint_object, descriptor_objectInMethod);

	Mat temp_descriptor;
	double dist_ham;
	int threshold = 100;
	int counter(0);

	for (int i = 0; i < descriptors.rows; i++){
		temp_descriptor = descriptors.row(i);
		dist_ham = norm(temp_descriptor, descriptor_objectInMethod, NORM_HAMMING);
//		cout << "\n Distance is : " << dist_ham << endl;

		if (dist_ham <= threshold)
		{
//			cout << "Distance is : " << dist_ham << endl;
			addStaticDictionary(temp_descriptor);
			counter++;
		}
	}
	cout << "\n Counter is : " << counter << endl;
	object = objectInitialLocation;
	descriptor_object = descriptor_objectInMethod;
}

void FeatureTracking::addStaticDictionary(Mat descriptor_word){
	int staticDictionarySize = static_dictionary.rows;
	if (staticDictionarySize >= StaticDictionarySize)
	{
//		cout << "Static Dictionary is Full" << endl;
		return;
	}
	static_dictionary.push_back(descriptor_word);
	return;
}

void FeatureTracking::searchObject(Mat img)
{

	int threshold = 100;
	int threshold2 = 250;
	Point2f points;
	KeyPoint kp_temp;
	vector<KeyPoint> keypoints;

	int alpha = 1;
	int r = searchRadious;
	int k = r / 2;
	float sigma2 = (float)r / 3.0;

	int x = object.x;
	int y = object.y;


	float deltaX(0), deltaY(0);
	for (int i = -r; i <= r; i++)
	{
		for (int j = -r; j <= r; j++)
		{
			if ((i<k) && (i>-k))
				deltaX = 0;
			if (i >= k)
				deltaX = alpha * (i - k);
			if (i <= -k)
				deltaX = alpha * (k + i);
			if ((j<k) && (j>-k))
				deltaY = 0;
			if (j >= k)
				deltaY = alpha * (j - k);
			if (j <= -k)
				deltaY = alpha * (k + j);
			x = object.x + (i + deltaX);
			y = object.y + (j + deltaY);
			points.x = x;
			points.y = y;
			kp_temp.pt.x = x;
			kp_temp.pt.y = y;
			kp_temp.size = 1;      // 7 and 1
			kp_temp.angle = -1;
			keypoints.push_back(kp_temp);
		}
	}




	Vec3b pixel;

	pixel[0] = 250;
	pixel[1] = 0;
	pixel[2] = 250;

	Mat img_searchGrid = img.clone();

	if (verbose){
		for (int i = 0; i < keypoints.size(); i++){
			img_searchGrid.at<uchar>(keypoints[i].pt.y, keypoints[i].pt.x) = 0;
		}

		imshow("Search Grid", img_searchGrid);
		waitKey(1);
	}
//	BriefDescriptorExtractor extractor(64);
//	BRISK extractor(30, 3, 1.5f);
//	FREAK extractor(true, true, patternScale, 4);
	Mat descriptors, temp_Descriptor;
	extractor.compute(img, keypoints, descriptors);

	double dist_ham;
	int min_hammingDistance(512);
	float score;
	float min_score(512);
	int index;
	float LocalPenalty;

	float savedLocal;
	int savedHamming;
	float savedScore;

	int winner = 0;

	//	fstream SaveDist;
	//	SaveDist.open("Scores.txt", ios::app);

	
	if (matcher.empty()){
		matcher.clear();
		matcher.add(vector<Mat>(1, descriptors));
	}
	else
	{
		matcher.clear();
		matcher.add(vector<Mat>(1, descriptors));
	}


	vector<DMatch> matches;
	matcher.match(static_dictionary, descriptors, matches);
//	vector<vector<DMatch>> matches;
//	matcher.knnMatch(static_dictionary, descriptors, matches, 2);


	//choose good matches
	vector<DMatch> good_matches;

	for (size_t i = 0; i < matches.size(); i++) 
	{
		
		dist_ham = matches[i].distance;
		if (dist_ham <= threshold)
			addStaticDictionary(descriptors.row(i));

		LocalPenalty = alpha * weightingFactor((keypoints[matches[i].trainIdx].pt.x - object.x), (keypoints[matches[i].trainIdx].pt.y - object.y), sigma2);
		score = dist_ham + LocalPenalty;

		if (score < min_score){
			//			if ( (dist_ham < min_hammingDistance) && (score < 50)){
			//				min_hammingDistance = dist_ham;
			savedHamming = dist_ham;
			savedLocal = LocalPenalty;
			savedScore = score;

			min_score = score;
			index = matches[i].trainIdx;
		}

		winner = 1;

		//if (matches[i].distance < minDistScore) {
		//	good_matches.push_back(matches[i]);
		//	minDistScore = matches[i].distance;
		//	index = i;
		//}
	}
	

	//for (int i = 0; i < descriptors.rows; i++)
	//{
	//	temp_Descriptor = descriptors.row(i);
	//	dist_ham = norm(temp_Descriptor, descriptor_object, NORM_HAMMING);
	//	if (dist_ham <= threshold)
	//		addStaticDictionary(temp_Descriptor);
	//	for (int j = 0; j < static_dictionary.rows; j++)
	//	{
	//		dist_ham = norm(temp_Descriptor, static_dictionary.row(j), NORM_HAMMING);

	//		//			if (dist_ham <= threshold2 )
	//		{

	//			LocalPenalty = alpha * weightingFactor((keypoints[i].pt.x - object.x), (keypoints[i].pt.y - object.y), sigma2);
	//			//				if (LocalPenalty < )
	//			score = dist_ham + LocalPenalty;

	//			//			cout << "Score is : " << score << endl;
	//			//			SaveDist << score << endl;


	//			if (score < min_score){
	//				//			if ( (dist_ham < min_hammingDistance) && (score < 50)){
	//				//				min_hammingDistance = dist_ham;
	//				savedHamming = dist_ham;
	//				savedLocal = LocalPenalty;
	//				savedScore = score;

	//				min_score = score;
	//				index = i;
	//			}
	//			winner = 1;
	//		}
	//		//			else{
	//		//				winner = 0;
	//		//			}
	//	}
	//}

	if (verbose)
		cout << "\n Hamming : " << savedHamming << "\tLocal : " << (int)savedLocal << "\tMin score : " << (int)savedScore << "\tindex : " << index;

	if (winner == 1){
		object.x = keypoints[index].pt.x;
		object.y = keypoints[index].pt.y;
		//		cout << "\nWinner";
	}

	
	Point2f center;	center.x = object.x; center.y = object.y;

	if (verbose){
		circle(img, center, 1, CV_RGB(250, 0, 0), 2, 8, 0);
		circle(img, center, patternScale, CV_RGB(250, 0, 0), 2, 8, 0);

		imshow("Located Object", img);
		waitKey(1);
	}

	//	SaveDist.flush();
	// 	SaveDist.close();

}

FeatureTracking::FeatureTracking(){
	
}

FeatureTracking::~FeatureTracking(){

};

int main(int argc, char* argv[]){

	/*******************************************************************************/
	//  First reading all of images in to ram (Images are from 0 to 841 )          //
	/*******************************************************************************/
	Mat imagesRam[1000], img_object, img_scene, img_scene_ROI;
	Mat temp;
	string imgName;
	
	int padding = (patternScale + searchRadious); // ? patternScale : searchRadious;
	for (int i = 0; i < 888; i++)
	{
		imgName = "images\\faceocc\\imgs\\img";					//  for surfs this is : imgName = "images\\surfer\\imgs\\img";
		if (i < 10)				imgName = imgName + "0000" + to_string(i) + ".png";
		else if (i < 100)		imgName = imgName + "000" + to_string(i) + ".png";
		else					imgName = imgName + "00" + to_string(i) + ".png";
		imagesRam[i] = imread(imgName, CV_LOAD_IMAGE_GRAYSCALE);
//		cv::Mat img(100, 100, CV_8UC3);
//		cv::Mat padded;
//		int padding = 20;
//		padded.create(imagesRam[i].rows+2*padding, imagesRam[i].cols+2*padding, imagesRam[i].type());
//		padded.setTo(cv::Scalar::all(0));
//		imagesRam[i].copyTo(padded(Rect(padding, padding, imagesRam[i].rows, imagesRam[i].cols)));
//		imagesRam[i] = padded.clone();
		Mat img = imagesRam[i].clone();
		Mat padded;
		padded.create(img.rows + 2 * padding, img.cols + 2 * padding, img.type());
		padded.setTo(cv::Scalar::all(0));
		Mat dst_roi = padded(Rect(padding, padding, img.cols, img.rows));
		img.copyTo(dst_roi);
		imagesRam[i] = padded.clone();
	}

	cout << "Reading Images into Ram Completed successfully ... \n\nPress any key ...\n";
	_getch();
	cout << "\nTracking in progress...\n";


	FeatureTracking tracker;
	tracker.initialTracker(imagesRam[0], Point2f(172 + padding, 144 + padding));   
	/*****		176, 139     172, 164		*****/
	/*****		for surf images this is : tracker.initialTracker(imagesRam[401], Point2f(286, 148));	*****/


	double start = getTickCount();
	for (int i = 0; i < 888; i++){

		tracker.searchObject(imagesRam[i]);

	}

	double finish = getTickCount();
	double duration = double(finish - start) / (double)getTickFrequency();

	cout << "\nDuration Time is : " << duration << endl;
	cout << "\nFrame Rate is : " << 888 / duration << " Frame/Sec" << endl;

	cout << "\nTracking Finished." << endl;
	_getch();
	return EXIT_SUCCESS;
}