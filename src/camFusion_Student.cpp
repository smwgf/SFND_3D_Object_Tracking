
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 10);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 2);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

float calDistance(cv::Point2f& pf, cv::Point2f& cf)
{
    float x = pf.x-cf.x;
    float y = pf.y-cf.y;
    return sqrt(x*x+y*y);
}
// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{    
    //loop for all matches
    for(cv::DMatch& match : kptMatches)
    {
        //if current box's roi does not contain match point, continue loop
        if(!boundingBox.roi.contains(kptsCurr[match.trainIdx].pt)) continue;
        if(calDistance(kptsCurr[match.trainIdx].pt,kptsPrev[match.queryIdx].pt) <10)
        {
            boundingBox.kptMatches.push_back(match);
        }          
    }
    cout << "##### cluster kp match ##### match size : " << boundingBox.kptMatches.size() << endl;
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{

    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // compute camera-based TTC from distance ratios
    //double meanDistRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size();    
    double dT = 1 / frameRate;
    //TTC = -dT / (1 - meanDistRatio);

    // STUDENT TASK (replacement for meanDistRatio)
    //double medianDistRatio = median(distRatios);
    //TTC = -dT / (1 - medianDistRatio);

    //Solution
    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence
    TTC = -dT / (1 - medDistRatio);
    cout << "##### TTC with Camera ##### TTC : " << TTC << " s" << endl;
}

/*
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // auxiliary variables
    double dT = 1.0/frameRate;        // time between two measurements in seconds
    double laneWidth = 4.0; // assumed width of the ego lane

    // find closest distance to Lidar points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        
        if (abs(it->y) <= laneWidth / 2.0)
        { // 3D point within ego lane?
            minXPrev = minXPrev > it->x ? it->x : minXPrev;
        }
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {

        if (abs(it->y) <= laneWidth / 2.0)
        { // 3D point within ego lane?
            minXCurr = minXCurr > it->x ? it->x : minXCurr;
        }
    }

    // compute TTC from both measurements
    TTC = minXCurr * dT / (minXPrev - minXCurr);
}
*/
/*
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // ...
        // auxiliary variables
    static double averagePrev2 = 0.0;
    double dT = 1.0f/frameRate;        // time between two measurements in seconds
    double laneWidth = 4.0; // assumed width of the ego lane

    // find closest distance to Lidar points within ego lane
    //double minXPrev = 1e9, minXCurr = 1e9;
    double averagePrev = 0.0 , averageCurr = 0.0, sumPrev = 0.0, sumCurr = 0.0;
    int cntPrev = 0, cntCurr = 0;
    std::vector<int> idxPrev=Ransac3DPlane(lidarPointsPrev,4000,0.02);
    for (auto it = idxPrev.begin(); it != idxPrev.end(); ++it)
    {
        double y = lidarPointsPrev[*it].y;                
        
        if (abs(y) <= laneWidth / 2.0)
        { // 3D point within ego lane?
            double x = lidarPointsPrev[*it].x;
            //minXPrev = minXPrev > x ? x : minXPrev;
            sumPrev+=x;
            cntPrev++;
        }
    }
    averagePrev = sumPrev/(double)cntPrev;
    std::vector<int> idxCurr=Ransac3DPlane(lidarPointsCurr,4000,0.02);
    for (auto it = idxCurr.begin(); it != idxCurr.end(); ++it)
    {
        double y = lidarPointsCurr[*it].y;
        if (abs(y) <= laneWidth / 2.0)
        { // 3D point within ego lane?
            double x = lidarPointsCurr[*it].x;
            //minXCurr = minXCurr > x ? x : minXCurr;
            sumCurr+=x;
            cntCurr++;            
        }
    }
    averageCurr = sumCurr/(double)cntCurr;
    // compute TTC from both measurements
    //TTC = minXCurr * dT / (minXPrev - minXCurr);
    double TTCwith2Times = 1e9;
    TTC = averageCurr * dT / (averagePrev - averageCurr);
    if(averagePrev2!=0.0)
    {
        TTCwith2Times = averageCurr * dT*2.0 / (averagePrev2 - averageCurr);
    }
    averagePrev2=averagePrev;
    //cout << "##### Test TTC #####" << endl;
    //cout << "## TTC : "<< TTC << " s" << endl;
    if(TTC>TTCwith2Times)
    {
        TTC= TTCwith2Times;
    }    
    //cout << "## TTC 2Times : "<< TTCwith2Times << " s" << endl;
    //cout << "## averagePrev : "<< averagePrev << " m" << endl;
    //cout << "## averageCurr : "<< averageCurr << " m" << endl;
}
*/
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // ...
    double dT = 1.0/frameRate;        // time between two measurements in seconds
    double laneWidth = 4.0; // assumed width of the ego lane

    // find closest distance to Lidar points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;
    double medXPrev = 1e9, medXCurr = 1e9;
    std::vector<double> vecPrev, vecCurr;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        
        if (abs(it->y) <= laneWidth / 2.0)
        { // 3D point within ego lane?
            vecPrev.push_back(it->x);
            minXPrev = minXPrev > it->x ? it->x : minXPrev;
        }
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {

        if (abs(it->y) <= laneWidth / 2.0)
        { // 3D point within ego lane?
            vecCurr.push_back(it->x);
            minXCurr = minXCurr > it->x ? it->x : minXCurr;
        }
    }
    std::sort(vecPrev.begin(), vecPrev.end());
    long medIndex = floor(vecPrev.size() / 2.0);
    medXPrev = vecPrev.size() % 2 == 0 ? (vecPrev[medIndex - 1] + vecPrev[medIndex]) / 2.0 : vecPrev[medIndex]; // compute median dist. ratio to remove outlier influence

    std::sort(vecCurr.begin(), vecCurr.end());
    medIndex = floor(vecCurr.size() / 2.0);
    medXCurr = vecCurr.size() % 2 == 0 ? (vecCurr[medIndex - 1] + vecCurr[medIndex]) / 2.0 : vecCurr[medIndex]; // compute median dist. ratio to remove outlier influence

    // compute TTC from both measurements
    TTC = minXCurr * dT / (minXPrev - minXCurr);
    TTC = medXCurr * dT / (medXPrev - medXCurr);
    //cout << "## TTC 2Times : "<< TTCwith2Times << " s" << endl;
    //cout << "## averagePrev : "<< averagePrev << " m" << endl;
    //cout << "## averageCurr : "<< averageCurr << " m" << endl;
}
/*
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    //loop for previous frames
    for(int i =0; i < prevFrame.boundingBoxes.size(); i++)
    {
        BoundingBox& boundingBox = prevFrame.boundingBoxes[i];
        //counting matched box in current frames
        std::vector<int> matchCount(currFrame.boundingBoxes.size(),0);
        //loop for all matches
        for(cv::DMatch& match : matches)
        {
            //if previous box's roi does not contain match point, continue loop
            if(!boundingBox.roi.contains(prevFrame.keypoints[match.queryIdx].pt)) continue;
            for(int j =0; j < currFrame.boundingBoxes.size(); j++)
            {
                BoundingBox& curBox = currFrame.boundingBoxes[j];
                //if current box's roi contains match point, increase match count
                if(curBox.roi.contains(currFrame.keypoints[match.trainIdx].pt))
                {
                    matchCount[j] = matchCount[j]+1;
                }
            }            
        }
        int max_index = max_element(matchCount.begin(),matchCount.end())-matchCount.begin();
        //if max matched count is not zero
        if(matchCount[max_index]>0)
        {
            bbBestMatches.insert(make_pair(i,max_index));
        }        
    }    
}
*/

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    //loop for current frames
    for(int i =0; i < currFrame.boundingBoxes.size(); i++)
    {
        BoundingBox& boundingBox = currFrame.boundingBoxes[i];
        //counting matched box in previouse frames
        std::vector<int> matchCount(prevFrame.boundingBoxes.size(),0);
        //loop for all matches
        for(cv::DMatch& match : matches)
        {
            //if current box's roi does not contain match point, continue loop
            if(!boundingBox.roi.contains(currFrame.keypoints[match.trainIdx].pt)) continue;
            for(int j =0; j < prevFrame.boundingBoxes.size(); j++)
            {
                BoundingBox& curBox = prevFrame.boundingBoxes[j];
                //if previous box's roi contains match point, increase match count
                if(curBox.roi.contains(prevFrame.keypoints[match.queryIdx].pt))
                {
                    matchCount[j] = matchCount[j]+1;
                }
            }            
        }
        int max_index = max_element(matchCount.begin(),matchCount.end())-matchCount.begin();
        //if max matched count is not zero
        if(matchCount[max_index]>0)
        {
            bbBestMatches.insert(make_pair(max_index,i));
        }        
    }    
}

std::vector<int> Ransac3DPlane(std::vector<LidarPoint> &cloud, int maxIterations, float distanceTol)
{
	std::vector<int> inliersResult;
	srand(time(NULL));
	
	// TODO: Fill in this function

	
	std::vector<int>* mostInlier (new std::vector<int>());
	

	// For max iterations 
	for(int i =0 ; i<maxIterations;i++)
	{
		std::vector<int>* currentInlier (new std::vector<int>());
		// Randomly sample subset and fit line
		int index1 = rand()%cloud.size();
		int index2, index3;
		do{
			index2 = rand()%cloud.size();
		}
		while(index1==index2);
		do{
			index3 = rand()%cloud.size();
		}
		while(index1==index3||index2==index3);
		
		float x1 =cloud[index1].x;
		float x2 =cloud[index2].x;
		float x3 =cloud[index3].x;
		float y1 =cloud[index1].y;
		float y2 =cloud[index2].y;
		float y3 =cloud[index3].y;
		float z1 =cloud[index1].z;
		float z2 =cloud[index2].z;
		float z3 =cloud[index3].z;
		float A = (y2-y1)*(z3-z1)-(z2-z1)*(y3-y1);
		float B = (z2-z1)*(x3-x1)-(x2-x1)*(z3-z1);
		float C = (x2-x1)*(y3-y1)-(y2-y1)*(x3-x1);
		float D = - A*x1 - B*y1 - C*z1;
		float sqrA = A*A;
		float sqrB = B*B;
		float sqrC = C*C;

		// Measure distance between every point and fitted line
		for(int j=0 ; j <cloud.size();j++)
		{			
			float x =cloud[j].x;
			float y =cloud[j].y;
			float z =cloud[j].z;
			float d = fabs(A*x+B*y+C*z +D)/sqrt(sqrA+sqrB+sqrC);
			// If distance is smaller than threshold count it as inlier
			if(d<distanceTol)
			{
				currentInlier->push_back(j);
			}			
		}
		if(mostInlier->size()<currentInlier->size())
		{
			delete mostInlier;
			mostInlier=currentInlier;
		}
		else
		{
			delete currentInlier;
		}
		

	}
	// Return indicies of inliers from fitted line with most inliers
	return *mostInlier;
}

std::vector<LidarPoint> filterCloud(std::vector<LidarPoint> &lidarPoints)
{
    std::vector<LidarPoint> buffer;    
    std::vector<int> idx=Ransac3DPlane(lidarPoints,2000,0.03);
    for (auto it = idx.begin(); it != idx.end(); ++it)
    {
        buffer.push_back(lidarPoints[*it]);
    }
    return buffer;
}