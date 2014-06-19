#include <iostream>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
#include <math.h> 

#include <sstream>
#include <string>
#include <vector>
#include <map>

using namespace std;
string _trainData;
string _testData;
string _modelFile;
string _predictResult;
string _evaluateResult;

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
	printf("\nloadData().....\n");
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
		
		//printf("parse file %d, calssId = %d, words.size = %d\n", i, classId, words.size());
	}

	map<int,int>::iterator it = _classFreq.begin();
	for(;it!=_classFreq.end();++it){
		printf("classId %d:	%d\n", it->first, it->second);
	}

	printf("wordDict size = %d\n", _wordDict.size());
	fin.close();
}

void computeModel(){
	printf("\ncomputeModel.....\n");
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
		//printf("classId = %d\n", classId);
		//printf("_clsWordFreq[%d].size = %d\n", classId, _clsWordFreq[classId].size());
		for(map<int,int>::iterator it2 = _clsWordFreq[classId].begin(); it2 != _clsWordFreq[classId].end(); ++it2){
			sum += it2->second;
		}
		//printf("_clsWordFreq[%d] words count = %f\n", classId, sum);	
		
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
	printf("\nsaveModel.....\n");
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
	printf("\nloadModel.....\n");
	_wordDict.clear();
	_classProb.clear();
	_clsWordProb.clear();

	ifstream fin(_modelFile.c_str());
	string sline;
	getline(fin, sline);
	//printf("sline = %s\n", sline.c_str());
	
	vector<string> items = split(sline, ' ');
	if(items.size() < 6){
		printf("Modle format error!\n");
	}
	int len = items.size();
	for(int i=0; i < len; ){
		int classId = atoi(items[i].c_str());
		if(++i >= len){
			printf("Model format error!\n");
			fin.close();
			return;
		}
		_classProb[classId] = atof(items[i].c_str());
		if(++i >= len){
			printf("Model format error!\n");
			fin.close();
			return;
		}
		_clsDefaultProb[classId] = atof(items[i].c_str());
		++i;
		printf("_classProb[%d] = %lf, _clsDefaultProb[%d] = %lf\n", classId, _classProb[classId], classId, _clsDefaultProb[classId]);
	}

	for(map<int,double>::iterator it = _classProb.begin(); it != _classProb.end(); ++it){
		int classId = it->first;
		_clsWordProb[classId] = map<int, double>();
		getline(fin, sline);
		items = split(sline, ' ');
		len = items.size();
		for(int i = 0; i<len; ){
			int wid = atoi(items[i].c_str());
			if(_wordDict.find(wid) == _wordDict.end()){
				_wordDict[wid] = 1;
			}
			else{
				_wordDict[wid] += 1;
			}
		
			i++;
			if(i >= len){
				printf("Model format error!\n");
				fin.close();
				return;
			}
			_clsWordProb[classId][wid] = atof(items[i].c_str());
			i++;	
			//printf("classId = %d, wid = %d, prob = %lf\n", classId, wid, _clsWordProb[classId][wid]);	
		}
	}
	printf("_wordDict size = %d\n", _wordDict.size());
	fin.close();
}

void predict(){
	printf("\npredict.....\n");
	std::vector<int> trueLabelList;
	std::vector<int> preLabelList;

	ifstream fin(_testData.c_str());
	ofstream out(_predictResult.c_str());

	std::vector<string> items;
	string sline;  
    while(getline(fin, sline)){  
    	int pos = sline.find("#");
		if(pos > 0){
			sline = sline.substr(0, pos);
		}
		vector<string> items = split(sline, ' ');
		if(items.size() < 1){
			printf("Test data format error!");
			continue;
		}

		int trueClassId = atoi(items[0].c_str());
		int preClassId = 0;
		double maxScore = 0.0f;
		double curScore = 0.0f;

		int j = 0;
		
		map<int, double> scoreDic;
		for(map<int,double>::iterator it = _classProb.begin(); it!=_classProb.end(); ++it){
			int classId = it->first;
			curScore = log(it->second);
			//printf("scoreDic[%d] = %lf, it->second = %lf\n", it->first, scoreDic[classId], it->second);
			for(int i = 1; i < items.size(); i++){
				int wid = atoi(items[i].c_str());
				if(_wordDict.find(wid) == _wordDict.end()){
					continue;
				}
				if(_clsWordProb[classId].find(wid) == _clsWordProb[classId].end()){
					curScore += log(_clsDefaultProb[classId]);
				}
				else{
					curScore += log(_clsWordProb[classId][wid]);
				}
			}

			if(j == 0 ){
				maxScore = curScore;
				preClassId = classId;
			}
			else if(curScore > maxScore){
				maxScore = curScore;
				preClassId = classId;
			}

			j++;
		}

		//printf("preClassId = %d, maxScore=%lf\n", preClassId, maxScore);

		trueLabelList.push_back(trueClassId);
		preLabelList.push_back(preClassId);
		out << trueClassId;
		out << " ";
		out << preClassId;
		out << "\n";
    }
    fin.close();
    out.close();
}

void evaluate(){
	printf("\nevaluate.....\n");
	ifstream fin(_predictResult.c_str());
	ofstream out(_evaluateResult.c_str());

	std::vector<int> trueLabelList;
	std::vector<int> preLabelList;
	float accuracy;
	std::map<int, float> precisonDic;
	std::map<int, float> recallDic;

	string sline;
	std::vector<string> items;

	//准确率(Accuracy)
	//(C11 + C22) / (C11 + C12 + C21 + C22)
	int total = 0;
	int correct = 0;
    while(getline(fin, sline)){ 
    	items = split(sline, ' ');
    	if(items.size() != 2){
    		printf("_predict.result format error!");
    		fin.close();
    		out.close();
    		return;
    	}
    	int trueClassId = atoi(items[0].c_str());
    	int preClassId = atoi(items[1].c_str());
    	if(trueClassId == preClassId){
			correct++;
    	}
    	total++;

    	trueLabelList.push_back(trueClassId);
    	preLabelList.push_back(preClassId);
    }
    //printf("correct = %d, total = %d\n", correct, total);

    accuracy = correct * 1.0f / total;

    //针对Yi的评估
    for(map<int,double>::iterator it = _classProb.begin(); it != _classProb.end(); ++it){
    	int classId = it->first;
    	total = 0;
    	correct = 0;

    	//精确率(Precision, Yi), 公式：C11/(C11+C21), 找对Yi的概率
    	for ( int i = 0 ; i < preLabelList.size() ; i++ )
    	{
    		if(preLabelList[i] != classId){
    			continue;
    		}
       		if(preLabelList[i] == trueLabelList[i]){
       			correct++;
       		}
       		total++;
   		}
   		precisonDic[classId] = correct * 1.0 / total;

    	//召回率(Recall， Yi)， 公式 C11/(C11 + C12)，找出Yi的概率
    	total = 0;
    	correct = 0;
    	for ( int i = 0 ; i < trueLabelList.size() ; i++ )
    	{
    		if(trueLabelList[i] != classId){
    			continue;
    		}
       		if(preLabelList[i] == trueLabelList[i]){
       			correct++;
       		}
       		total++;
   		}
   		recallDic[classId] = correct * 1.0 / total;
    }


    printf("accuracy = %f\n", accuracy);

    out << "accuracy:\n";
    out << accuracy;
    out << "\n\n";

    out << "precison:\n";
    for(map<int, float>::iterator it = precisonDic.begin(); it != precisonDic.end(); ++it){
    	printf("precison[%d] = %f\n", it->first, it->second);
    	out << it->first;
    	out <<" ";
    	out << it->second;
    	out << "\n";
    }
    
    out << "\n";
    out << "precall:\n";
    for(map<int, float>::iterator it = recallDic.begin(); it != recallDic.end(); ++it){
    	printf("recall[%d] = %f\n", it->first, it->second);
    	out << it->first;
    	out <<" ";
    	out << it->second;
    	out << "\n";
    }

	fin.close();
	out.close();
}

int main(int argc, char **argv) {
	_trainData = "../data.train";
	_testData = "../data.test";
	_modelFile = "../data.model";
	_predictResult = "../predict.result";
	_evaluateResult = "../evaluate.result";


	loadData();
	computeModel();
	saveModel();

	//
	loadModel();
	predict();
	evaluate();

	return 0;
}

