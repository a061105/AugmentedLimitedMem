#ifndef SAMPLE_MODEL_H
#define SAMPLE_MODEL_H

#include <algorithm>
#include <iostream>
#include <set>
#include <stdlib.h>
#include <fstream>

class SampleModel{
	
	public:
	
	time_t ramp_start;
	double last_time;
	double sample_duration;
	double cumul_writetime;
	char* model_dir;
	double* model;
	
	int count_same;
	double end_time;
	Subsolver* solver;

	int iter;

	SampleModel( char* _model_dir  , double dur, double _end_time, Subsolver* _solver){
		
		model_dir = _model_dir;
		end_time = _end_time;
		solver = _solver;
		
		char* cmd1 = new char[1024];
		sprintf(cmd1, "rm -f %s/*", model_dir);
		system(cmd1);
		
		char* cmd2 = new char[1024];
		sprintf(cmd2, "mkdir -p %s", model_dir);
		system(cmd2);

		sample_duration = dur;
		last_time = -sample_duration - 10000;
		count_same =0;
	}
	
	void start(){
		
		time(&ramp_start);
		cumul_writetime = 0.0;
	}

	virtual void setModel(double* s_model){
		model = s_model;
	}
	
	virtual void writeModel(char* fname){
		
		solver->writeModel(fname, model);
	}
	
	void sample()
	{
		//modelname is the directory name
		
		if(model_dir != NULL) 
		{
			time_t now;
			time(&now);
			double duration = difftime(now, ramp_start);
			
			if(duration > end_time){
				exit(0);
			}
			
			if( duration - last_time > sample_duration )
			{
				char* mn = new char[10000];
				if( duration != last_time ){
					sprintf(mn, "%s/%f", model_dir, duration-cumul_writetime);
					count_same=0;
				}else{
					count_same++;
					sprintf(mn, "%s/%f", model_dir, duration+count_same*0.2-cumul_writetime);
				}
				
				time_t write_start; time(&write_start);
				writeModel(mn);
				time_t write_end; time(&write_end); 
				cumul_writetime += difftime(write_end,write_start);
				delete[] mn;

				last_time = duration;
				//sample_duration *= 1.1;
			}
		}
	}

	void sample(int _iter)
	{
		//modelname is the directory name
		
		if(model_dir != NULL) 
		{
			time_t now;
			time(&now);
			double duration = difftime(now, ramp_start);
			iter = _iter;
			
			if(duration > end_time){
				exit(0);
			}
			
			if( duration - last_time > sample_duration )
			{
				char* mn = new char[10000];
				if( duration != last_time ){
					//sprintf(mn, "%s/%f", model_dir, duration-cumul_writetime);
					sprintf(mn, "%s/%d", model_dir, iter);
					count_same=0;
				}else{
					count_same++;
					//sprintf(mn, "%s/%f", model_dir, duration+count_same*0.2-cumul_writetime);
					sprintf(mn, "%s/%d", model_dir, iter);
				}
				
				time_t write_start; time(&write_start);
				writeModel(mn);
				time_t write_end; time(&write_end); 
				cumul_writetime += difftime(write_end,write_start);
				delete[] mn;

				last_time = duration;
				//sample_duration *= 1.1;
			}
		}
	}
};

class SampleWeight:public SampleModel{
	
	public:
	
	double* w;
	int d;
	
	SampleWeight( char* _model_dir  , double dur, double _endTime, Subsolver* subsolver ):SampleModel(_model_dir,dur,_endTime, subsolver)
	{

	}
	
	virtual void setModel(double* w, int d){
		this->w = w;
		this->d = d;
	}
	
	virtual void writeModel(char* fname){
		
		ofstream fout(fname);
		fout << "class: 1" << endl;
		fout << d << endl;
		for(int i=0;i<d;i++){
			if(w[i]!=0.0)
				fout << i+1 << ":" << w[i] << " ";
		}
		fout.close();
	}
};

#endif
