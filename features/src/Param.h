#pragma once
#ifndef  PARAM_H_
#define  PARAM_H_
#include <string>
#include <filesystem>
#include <iostream>
using namespace std;

string pol = "/data";
vector<string> pol_dataset = { "/hh" , "/vv" ,"/hv" ,"/vh"};

string ctElements = "/CTelememts";
string MP = "/MP";
string decomp = "/decomp";
string color = "/color";
string texture = "/texture";
string statistic = "/statistic";
vector<string> dataset_name = { "/feature" ,"/samplePoint" };



#endif