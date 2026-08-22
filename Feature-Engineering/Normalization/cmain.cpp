#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;

class Normalization{
  private:
    std::vector<double> test_input_list;
  public:
    //CONSTRUCTOR
    Normalization(const std::vector<double>& lst) : test_input_list(lst) {}  //using const allow only to read not edit

    vector<double> min_maxScaling(){
        vector<double> scaledResult;
        double maxVal = test_input_list[0];
        double minVal = test_input_list[0];

        if (test_input_list.empty()) return scaledResult; 
        for(double i : test_input_list){
            maxVal = max(i , maxVal);
            minVal = min(i , minVal);
        }
        double tempVal = maxVal - minVal;
        for(double j : test_input_list){
            scaledResult.push_back((j - minVal) / tempVal);
        }
        return scaledResult;
    }

    vector<double> meanScaling(){
        vector<double> scaledResult;
        double maxVal = test_input_list[0];
        double minVal = test_input_list[0];

        if (test_input_list.empty()) return scaledResult; 

        double totalSum =0;

        for(double i : test_input_list){
            maxVal = max(i , maxVal);
            minVal = min(i , minVal);
            totalSum = totalSum + i;
        }
        double mean = totalSum/test_input_list.size();
        double tempVal = maxVal - minVal;
        for(double j : test_input_list){
            scaledResult.push_back((j - mean) / tempVal);
        }

        return scaledResult;
    }

    vector<double> maxAbsolute_Scaling(){ //useful in sparse data
        vector<double> scaledResult;
        double maxVal = test_input_list[0];

        if (test_input_list.empty()) return scaledResult; 

        for(double i : test_input_list){
            maxVal = max(i , maxVal);
        }

        if(maxVal < 0){ //-ve value
            maxVal = maxVal * (-1);
        }

        for(double j : test_input_list){
            scaledResult.push_back(j / maxVal);
        }
        return scaledResult;
    }

    vector<double> robustScaling(){
        std::sort(test_input_list.begin(), test_input_list.end()); 
        vector<double> scaledResult;
        double median = 0;

        if(test_input_list.size() % 2 != 0){ //for odd
            double tempVal = (test_input_list.size() -1) / 2;
            median = test_input_list[tempVal];
        }else{
            double tempVal2 = test_input_list.size()/ 2;
            double tempVal3 = test_input_list[tempVal2] + test_input_list[tempVal2 -1];
            median = tempVal3/2;
        }
         
        double IQR = test_input_list[std::round(3*(test_input_list.size())/4)]- test_input_list[std::round((test_input_list.size())/4)];
        
        for(double i : test_input_list){
            scaledResult.push_back((i - median)/IQR);
        }
        
        
        cout << IQR;
        
        return scaledResult;
    }
};

int main(){
    std::vector<double> test_data = {
    73.52, 14.89, 44.12, 92.05, 5.11, 67.84, 31.45, 88.91, 12.33, 53.67,
    99.14, 25.40, 61.22, 8.76, 47.93, 83.19, 36.54, 70.08, 19.65, 55.42,
    41.18, 96.73, 2.34, 64.51, 28.09, 85.66, 11.02, 50.88, 77.23, 22.91,
    58.70, 6.44, 49.33, 91.17, 33.82, 79.50, 15.26, 63.11, 87.04, 18.39,
    46.05, 94.21, 4.88, 68.12, 29.74, 82.59, 13.90, 52.41, 75.99, 21.06,
    40.23, 97.85, 1.15, 66.39, 26.47, 84.13, 9.88, 51.56, 78.42, 24.33,
    57.12, 7.89, 48.01, 90.62, 35.19, 81.34, 16.71, 62.45, 86.29, 17.04,
    43.88, 93.56, 3.22, 65.04, 30.61, 89.47, 10.55, 54.18, 76.81, 23.50,
    59.94, 5.72, 45.16, 95.12, 32.07, 80.22, 14.48, 60.77, 88.01, 19.92,
    42.61, 98.34, 0.45, 69.57, 27.83, 83.68, 11.74, 55.91, 74.25, 20.66
    };

    vector<double> output_data = Normalization(test_data).robustScaling();

    for(double i : output_data){
        cout << i << std::endl;
    }
    return 0;
}