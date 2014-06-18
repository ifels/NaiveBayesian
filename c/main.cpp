#include <iostream>
#include <fstream>

#include<stdio.h>
#include<stdlib.h>

#include <sstream>
#include <string>
#include <vector>
#include <map>

using namespace std;
string _trainData;
string _testData;
string _modelFile;

double DefaultFreq = 0.5;    //平滑参数

map<int, int> _wordDict;  //词典

map<int, int> _classFreq;		//分类的频度, Count(Yi), map<classId, classFreq>
map<int, double> _classProb;	//分类的频率， P(Yi), map<classId, classProb>

//Num(Xi, Yi)
map<int, map<int, int> > _clsWordFreq; //分类内各词的频度, map<calssId, map<wid, count> >
map<int , map<int, double> > _clsWordProb; //分类内各词的频率, map<calssId, map<wid, wordProb> >
map<int, double> _clsDefaultProb;  //某分类内没有出现的词的缺省概率, map<classId, defaultProb> 

vector<string> split(const string &s, char delim) {
    vector<string> elems;
	stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    
    return elems;
}

void loadData() {
	ifstream fin(_trainData.c_str());  
    string sline;  
	int i = 0;
    while(getline(fin, sline)){   	
		int pos = sline.find("#");
		if(pos > 0){
			sline = sline.substr(0, pos);
		}
		vector<string> words = split(sline, ' ');
		if(words.size() < 1){
			printf("Format error!");
			break;
		}
		int classId = atoi(words[0].c_str());
		if(_classFreq.find(classId) == _classFreq.end()){
			_classFreq[classId] = 0;
			_classProb[classId] = 0.0;
			_clsWordFreq[classId] = map<int, int>();
			_clsWordProb[classId] = map<int, double>();
			_clsDefaultProb[classId] = 0.0;
		}
		_classFreq[classId] += 1;

		for ( int j = 1 ; j < words.size() ; j++ ){
			int wid = atoi(words[j].c_str());
			if(_wordDict.find(wid) == _wordDict.end()){
				_wordDict[wid] = 1;
			}
			else{
				_wordDict[wid] += 1;
			}
		
			if(_clsWordFreq[classId].find(wid) == _clsWordFreq[classId].end()){
				_clsWordFreq[classId][wid] = 1;
			}
			else{
				_clsWordFreq[classId][wid] += 1;
			}
		}
		i++;
		
		printf("parse file %d, calssId = %d, words.size = %d\n", i, classId, words.size());
	}

	map<int,int>::iterator it = _classFreq.begin();
	for(;it!=_classFreq.end();++it){
		printf("classId %d:	%d\n", it->first, it->second);
	}

	printf("wordDict size = %d\n", _wordDict.size());
}

void computeModel(){
	int sum1 = 0;

	//P(Yi)
	for(map<int,int>::iterator it = _classFreq.begin(); it!=_classFreq.end(); ++it){
		sum1 += it->second;
	}
	printf("total file count = %d\n", sum1);

	for(map<int,double>::iterator it = _classProb.begin(); it!=_classProb.end(); ++it){
		it->second = _classFreq[it->first] * 0.1 / sum1;
		printf("P(Y%d) = %f\n", it->first, it->second);
	}

	//P(Xi | Yj)
	for(map<int,int>::iterator it = _classFreq.begin(); it != _classFreq.end(); ++it){
		double sum = 0.0;
		int classId = it->first;
		printf("classId = %d\n", classId);
		printf("_clsWordFreq[%d].size = %d\n", classId, _clsWordFreq[classId].size());
		for(map<int,int>::iterator it2 = _clsWordFreq[classId].begin(); it2 != _clsWordFreq[classId].end(); ++it2){
			sum += it2->second;
		}
		printf("_clsWordFreq[%d] words count = %f\n", classId, sum);	
		
		//平滑	
		sum += _wordDict.size() * DefaultFreq;
		_clsDefaultProb[classId] = DefaultFreq / sum;
		printf("_clsDefaultProb[%d] = %lf\n", classId,  _clsDefaultProb[classId]);

		for(map<int,int>::iterator it2 = _clsWordFreq[classId].begin(); it2 != _clsWordFreq[classId].end(); ++it2){
			_clsWordProb[classId][it2->first] =  (it2->second + DefaultFreq) / sum;	
			//printf("_clsWordProb[%d][%d] = %f\n", classId, it2->first, _clsWordProb[classId][it2->first]);	
		}
	}
	
}

void saveModel(){
	ofstream out(_modelFile.c_str());

	//P(Yj)
	for(map<int,double>::iterator it = _classProb.begin(); it!=_classProb.end(); ++it){
		out << it->first;
		out << " ";
		out << it->second;
		out << " ";
		out << _clsDefaultProb[it->first];
		out << " ";
	}
	out << "\n";

	//P(Xi | Yj)
	for(map<int,double>::iterator it = _classProb.begin(); it != _classProb.end(); ++it){
		int classId = it->first;
		for(map<int,double>::iterator it2 = _clsWordProb[classId].begin(); it2 != _clsWordProb[classId].end(); ++it2){
			out << it2->first;
			out << " ";
			out << it2->second;
			out << " ";
		}
		out << "\n";
	}
	out.close();
}

void loadModel(){
	_wordDict.clear();
	_classProb.clear();
	_clsWordProb.clear();

	ifstream fin(_modelFile.c_str());
	string sline;
	getline(fin, sline);
	printf("sline = %s\n", sline.c_str());


}

int main(int argc, char **argv) {
	_trainData = "../data.train";
	_testData = "../data.test";
	_modelFile = "../data.model";
	loadData();
	computeModel();
	saveModel();

	//
	loadModel();
	return 0;
}


