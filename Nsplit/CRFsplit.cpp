#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include "Subsolver.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define MAX_LINE 10000000

const int PATH_LENGTH = 10240;

using namespace std;

char readstr[MAX_LINE]; 
int main(int argc, char **argv)
{
	//Parse Command Input 
	char input_file_name[PATH_LENGTH];
	char buf[PATH_LENGTH];
	char dir_name[PATH_LENGTH];
	char* solverName;
	int D, N, K, L;
	
	strcpy(input_file_name, argv[1]);
	sprintf(dir_name,"%s.%s",argv[1],argv[2] );
	K = atoi(argv[2]);
	N = atoi(argv[3]);
	D = atoi(argv[4]);
	L = atoi(argv[5]);
	solverName = argv[6];
	
	if(mkdir(dir_name,0755) != 0)
	{
		fprintf(stderr,"Cannot make dir %s; remove or rename the directoy if it exists\n", dir_name);
		return -1;
	}

	// Write Meta File
	sprintf(buf, "%s/meta", dir_name);
	FILE *metafile = fopen(buf, "w");
	fprintf(metafile,"%d %d %d %d\n",K, N, D, L);
	int i;
	char *pch1,datafilename[PATH_LENGTH];
	char temp_input_file_name[PATH_LENGTH];
	strcpy(temp_input_file_name, input_file_name);
	pch1 = strtok (temp_input_file_name,"/");
	while (pch1 != NULL)
	{
		strcpy(datafilename,pch1);
		pch1 = strtok (NULL, "/");
	}
	for(i = 0;i < K;i++)
	{
		fprintf(metafile,"%s.%d.bin\n",datafilename,i);
	}
	fclose(metafile);
	
	// Split original dataset into several K blocks

	char** dataFname = new char*[K];
	for(int i=0;i<K;i++)
		dataFname[i] = new char[PATH_LENGTH];
	
	FILE *datafile;
	sprintf(buf, "%s", input_file_name);
	FILE *input_fp = fopen(buf,"r");
	int now_instance = 0,total_num;
	fgets(readstr,MAX_LINE,input_fp);
	
	// write ASCII Data block 
	for(i = 0;i < K;i++)
	{
		sprintf(dataFname[i],"%s/%s.%d",dir_name,datafilename,i);
		datafile = fopen(dataFname[i],"w");
		if(i == K-1)
			total_num = N - now_instance;
		else
			total_num = N/K;
		
		int j = 0;
		while(j < total_num)
		{
			if(fgets(readstr,MAX_LINE,input_fp) == NULL)
				break;
			fputs(readstr,datafile);
			if(strcmp(readstr,"\n") == 0)
				j++;
		}
		now_instance += total_num;
		fclose(datafile);
	}
	
	//Create Subsolvers to read and save data (in binary)
	Subsolver* solver = create_subsolver(solverName, D, L);
	char dataBinaryFname[PATH_LENGTH];
	for(int i=0;i<K;i++){
		solver->readData(dataFname[i]);
		sprintf(dataBinaryFname, "%s.bin", dataFname[i]);
		solver->save(dataBinaryFname);
		solver->release();
		
		char cmd[1000];
		sprintf(cmd,"rm -f %s",dataFname[i]);
		system(cmd);
	}
	
	return 0;
}
