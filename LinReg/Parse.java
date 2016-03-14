import java.io.*;

class Parse{
	public static void main(String[] args){
			
		if( args.length < 2 ){
			System.err.println("Usage: java Parse [input] [output]");
			System.exit(0);
		}

		String input = args[0];
		String output = args[1];

		try{	
			BufferedReader bufr = new BufferedReader(new FileReader(input));
			BufferedWriter bufw = new BufferedWriter(new FileWriter(output));
			
			String line;
			String[] tokens;
			while( (line=bufr.readLine()) != null ){
				
				tokens = line.split(" ");
				bufw.write(tokens[0]+" ");
				for(int i=1;i<tokens.length;i++){
					
					bufw.write(tokens[i].split(":")[1]+" ");
				}
				bufw.newLine();
			}
			bufr.close();
			bufw.close();

		}catch(Exception e){
			e.printStackTrace();
			System.exit(0);
		}
	}
}
