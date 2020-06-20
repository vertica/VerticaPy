<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Movies-Scoring-and-Clustering">Movies Scoring and Clustering<a class="anchor-link" href="#Movies-Scoring-and-Clustering">&#182;</a></h1><p>This notebook is an example on how to use the Vertica ML Python Library. We will use the filmtv_movies dataset to evaluate the quality of the movies and create clusters based on similarity. You can download the Jupyter Notebook of the study by clicking <a href="movies.ipynb">here</a>. You can download the dataset of the study by clicking <a href = "../../data/filmtv_movies.csv">here</a>. We have access to the following variables.</p>
<ul class="ul_content">
    <li><b>year: </b>Movie Year of production</li>
    <li><b>filmtv_id: </b>Movie ID</li>
    <li><b>title: </b>Movie Title</li>
    <li><b>genre: </b>Movie Genre</li>
    <li><b>country: </b>Movie Country</li>
    <li><b>description: </b>Movie Description</li>
    <li><b>notes: </b>Information about the movie</li>
    <li><b>duration: </b>Movie Duration</li>
    <li><b>votes: </b>Number of votes</li>
    <li><b>avg_vote: </b>Averaged rate</li>
    <li><b>director: </b>Movie Director</li>
    <li><b>actors: </b>List of actors which played in the movie</li>

</ul><p>We will follow the entire Data Science cycle (Data Exploration - Data Preparation - Data Modeling - Model Evaluation - Model Deployment) and solve this easy use-case. The purpose is to show you some of the library interesting functionalities.</p>
<h2 id="Initialization">Initialization<a class="anchor-link" href="#Initialization">&#182;</a></h2><p>Let's create the Virtual DataFrames of the datasets.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python</span> <span class="k">import</span> <span class="o">*</span>
<span class="n">filmtv_movies</span> <span class="o">=</span> <span class="n">vDataFrame</span><span class="p">(</span><span class="s2">&quot;filmtv_movies&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">filmtv_movies</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">1</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>avg_vote</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>description</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>genre</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>votes</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>year</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>notes</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>title</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>filmtv_id</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>duration</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>country</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">Mickey Rourke, Steve Guttenberg, Ellen Barkin, Daniel Stern, Kevin Bacon, Timothy Daly, Paul Reiser, Kelle Kipp, Colette Blonigan</td><td style="border: 1px solid white;">7.20</td><td style="border: 1px solid white;">Five boys from Baltimore are in the habit of meeting periodically for dinner and playing tricks together. One of them, Boogie, is in trouble because of his passion for betting: he owes a gangster two thousand dollars and to find them he keeps betting and losing. The others have marital or sex-related problems, and everyone has a few jokes.</td><td style="border: 1px solid white;">Comedy</td><td style="border: 1px solid white;">Barry Levinson</td><td style="border: 1px solid white;">15</td><td style="border: 1px solid white;">1982.00</td><td style="border: 1px solid white;">A cast of will be famous for Levinson's directorial debut. Very bitter and very well written: jokes are seen as a manifestation of immaturity rather than carefree; Rourke enhances his charge of beautiful darkness. Guttenberg is the good guy in the group.</td><td style="border: 1px solid white;">Diner</td><td style="border: 1px solid white;">18</td><td style="border: 1px solid white;">95</td><td style="border: 1px solid white;">United States</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: filmtv_movies, Number of rows: 53397, Number of columns: 12
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Data-Exploration-and-Preparation">Data Exploration and Preparation<a class="anchor-link" href="#Data-Exploration-and-Preparation">&#182;</a></h2><p>Fidelizing customers can be very hard for any movies streaming platform. One of the biggest task is to find a good catalog of movies. Segmenting movies using different techniques can be a good idea to be able to propose to customers a complete films catalog.</p>
<p>First, let's explore the dataset.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[2]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">filmtv_movies</span><span class="o">.</span><span class="n">describe</span><span class="p">(</span><span class="n">method</span> <span class="o">=</span> <span class="s1">&#39;categorical&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>dtype</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>unique</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>count</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>top</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>top_percent</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"actors"</b></td><td style="border: 1px solid white;">varchar(2218)</td><td style="border: 1px solid white;">50121</td><td style="border: 1px solid white;">50372</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">5.665</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"avg_vote"</b></td><td style="border: 1px solid white;">numeric(6,2)</td><td style="border: 1px solid white;">89</td><td style="border: 1px solid white;">53397</td><td style="border: 1px solid white;">6.00</td><td style="border: 1px solid white;">15.014</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"description"</b></td><td style="border: 1px solid white;">varchar(2232)</td><td style="border: 1px solid white;">354</td><td style="border: 1px solid white;">359</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">99.328</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"genre"</b></td><td style="border: 1px solid white;">varchar(22)</td><td style="border: 1px solid white;">27</td><td style="border: 1px solid white;">53195</td><td style="border: 1px solid white;">Drama</td><td style="border: 1px solid white;">30.138</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"director"</b></td><td style="border: 1px solid white;">varchar(1066)</td><td style="border: 1px solid white;">19160</td><td style="border: 1px solid white;">53335</td><td style="border: 1px solid white;">Mario Mattòli</td><td style="border: 1px solid white;">0.137</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"votes"</b></td><td style="border: 1px solid white;">int</td><td style="border: 1px solid white;">588</td><td style="border: 1px solid white;">53397</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">22.988</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"year"</b></td><td style="border: 1px solid white;">numeric(8,2)</td><td style="border: 1px solid white;">111</td><td style="border: 1px solid white;">53387</td><td style="border: 1px solid white;">2016.00</td><td style="border: 1px solid white;">3.092</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"notes"</b></td><td style="border: 1px solid white;">varchar(1048)</td><td style="border: 1px solid white;">105</td><td style="border: 1px solid white;">106</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">99.801</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"title"</b></td><td style="border: 1px solid white;">varchar(486)</td><td style="border: 1px solid white;">50584</td><td style="border: 1px solid white;">53395</td><td style="border: 1px solid white;">Les Vampires</td><td style="border: 1px solid white;">0.019</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"filmtv_id"</b></td><td style="border: 1px solid white;">int</td><td style="border: 1px solid white;">53397</td><td style="border: 1px solid white;">53397</td><td style="border: 1px solid white;">18</td><td style="border: 1px solid white;">0.002</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"duration"</b></td><td style="border: 1px solid white;">int</td><td style="border: 1px solid white;">282</td><td style="border: 1px solid white;">53397</td><td style="border: 1px solid white;">90</td><td style="border: 1px solid white;">11.798</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"country"</b></td><td style="border: 1px solid white;">varchar(208)</td><td style="border: 1px solid white;">2394</td><td style="border: 1px solid white;">53346</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">41.141</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[2]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We can see too many missing values for the features description and notes (More than 99% for both of them). We can drop these two features.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">filmtv_movies</span><span class="o">.</span><span class="n">drop</span><span class="p">([</span><span class="s2">&quot;description&quot;</span><span class="p">,</span> <span class="s2">&quot;notes&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>avg_vote</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>genre</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>votes</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>year</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>title</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>filmtv_id</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>duration</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>country</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">Mickey Rourke, Steve Guttenberg, Ellen Barkin, Daniel Stern, Kevin Bacon, Timothy Daly, Paul Reiser, Kelle Kipp, Colette Blonigan</td><td style="border: 1px solid white;">7.20</td><td style="border: 1px solid white;">Comedy</td><td style="border: 1px solid white;">Barry Levinson</td><td style="border: 1px solid white;">15</td><td style="border: 1px solid white;">1982.00</td><td style="border: 1px solid white;">Diner</td><td style="border: 1px solid white;">18</td><td style="border: 1px solid white;">95</td><td style="border: 1px solid white;">United States</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">Gian Maria Volonté, Irene Papas, Gabriele Ferzetti, Salvo Randone, Laura Nucci, Mario Scaccia, Luigi Pistilli, Leopoldo Trieste</td><td style="border: 1px solid white;">7.80</td><td style="border: 1px solid white;">Drama</td><td style="border: 1px solid white;">Elio Petri</td><td style="border: 1px solid white;">102</td><td style="border: 1px solid white;">1967.00</td><td style="border: 1px solid white;">A ciascuno il suo</td><td style="border: 1px solid white;">22</td><td style="border: 1px solid white;">93</td><td style="border: 1px solid white;">Italy</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">Alan Steel, Mary Arden, Sergio Ciani, Ivano Davoli, Giovanna Galletti, Aïché Nana, Charlie Charun, Gilberto Mazzi</td><td style="border: 1px solid white;">6.50</td><td style="border: 1px solid white;">Thriller</td><td style="border: 1px solid white;">Ray Morrison (Angelo Dorigo)</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">1966.00</td><td style="border: 1px solid white;">A... come assassino</td><td style="border: 1px solid white;">24</td><td style="border: 1px solid white;">80</td><td style="border: 1px solid white;">Italy</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">George Hilton, Ennio Girolami, Marta Padovan, Alfonso De La Vega, Venancio Muro, Alfonso Rojas, Luis Marin</td><td style="border: 1px solid white;">5.50</td><td style="border: 1px solid white;">Adventure</td><td style="border: 1px solid white;">Leon Klimovsky</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">1968.00</td><td style="border: 1px solid white;">A Ghentar si muore facile</td><td style="border: 1px solid white;">30</td><td style="border: 1px solid white;">101</td><td style="border: 1px solid white;">Italy</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">Patsy Kensit, Stéphane Freiss, Mouss Diouf, Anne-Marie Pisani, Joseph Momo, Jean-Marc Truong, An Luu</td><td style="border: 1px solid white;">4.00</td><td style="border: 1px solid white;">Comedy</td><td style="border: 1px solid white;">Carol Wiseman</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1990.00</td><td style="border: 1px solid white;">Does This Mean We are Married?</td><td style="border: 1px solid white;">31</td><td style="border: 1px solid white;">90</td><td style="border: 1px solid white;">United States</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[3]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: filmtv_movies, Number of rows: 53397, Number of columns: 10</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We have access to more than 50000 movies with 27 different genres. Let's now look at the movies having the best rates.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">filmtv_movies</span><span class="o">.</span><span class="n">sort</span><span class="p">({</span><span class="s2">&quot;avg_vote&quot;</span> <span class="p">:</span> <span class="s2">&quot;desc&quot;</span><span class="p">})</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>avg_vote</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>genre</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>votes</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>year</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>title</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>filmtv_id</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>duration</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>country</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">Edwige Feuillère, Nicole Berger, Pierre-Michel Beck, Charles Dechamps, Louis De Funès</td><td style="border: 1px solid white;">10.00</td><td style="border: 1px solid white;">Drama</td><td style="border: 1px solid white;">Claude Autant-Lara</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1954.00</td><td style="border: 1px solid white;">Le blé en herbe</td><td style="border: 1px solid white;">22937</td><td style="border: 1px solid white;">106</td><td style="border: 1px solid white;">France</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">Francisco Rabal, Katy Jurado, Flor Eduarda Gurrola, Carolina Papaleo</td><td style="border: 1px solid white;">10.00</td><td style="border: 1px solid white;">Grotesque</td><td style="border: 1px solid white;">Arturo Ripstein</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1998.00</td><td style="border: 1px solid white;">El Evangelio de las Maravillas</td><td style="border: 1px solid white;">22639</td><td style="border: 1px solid white;">112</td><td style="border: 1px solid white;">Argentina, Mexico, Spain</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">George Nader, Maggie Smith, Bernard Lee</td><td style="border: 1px solid white;">10.00</td><td style="border: 1px solid white;">Drama</td><td style="border: 1px solid white;">Seth Holt</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1958.00</td><td style="border: 1px solid white;">Nowere to Go</td><td style="border: 1px solid white;">14016</td><td style="border: 1px solid white;">87</td><td style="border: 1px solid white;">Great Britain</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">Walter Huston, Karen Morley, Franchot Tone, Arthur Bryon, Dickie Moore</td><td style="border: 1px solid white;">10.00</td><td style="border: 1px solid white;">Drama</td><td style="border: 1px solid white;">Gregory La Cava</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1933.00</td><td style="border: 1px solid white;">Gabriel Over the White House</td><td style="border: 1px solid white;">13569</td><td style="border: 1px solid white;">87</td><td style="border: 1px solid white;">United States</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">Anthony Meyer, David Meyer, Helen Mirren, Quentin Crisp</td><td style="border: 1px solid white;">10.00</td><td style="border: 1px solid white;">Drama</td><td style="border: 1px solid white;">Celestino Coronado</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1976.00</td><td style="border: 1px solid white;">Hamlet</td><td style="border: 1px solid white;">23049</td><td style="border: 1px solid white;">65</td><td style="border: 1px solid white;">Great Britain</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[4]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: filmtv_movies, Number of rows: 53397, Number of columns: 10</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Some of the movies didn't get a good rate because of their quality. They just don't have enough votes. The rate can be significant when many spectators share their opinion about the movie. Let's consider the top 10 movies having at least 10 votes.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">filmtv_movies</span><span class="o">.</span><span class="n">search</span><span class="p">(</span><span class="n">conditions</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;votes &gt; 10&quot;</span><span class="p">]</span> <span class="p">,</span> 
                     <span class="n">order_by</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;avg_vote&quot;</span> <span class="p">:</span> <span class="s2">&quot;desc&quot;</span> <span class="p">},</span> 
                     <span class="n">limit</span> <span class="o">=</span> <span class="mi">10</span><span class="p">)</span> 
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>avg_vote</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>genre</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>votes</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>year</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>title</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>filmtv_id</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>duration</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>country</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">Marlon Brando, Al Pacino, Robert Duvall, James Caan, Diane Keaton</td><td style="border: 1px solid white;">9.80</td><td style="border: 1px solid white;">Drama</td><td style="border: 1px solid white;">Francis Ford Coppola</td><td style="border: 1px solid white;">33</td><td style="border: 1px solid white;">1992.00</td><td style="border: 1px solid white;">The Godfather Trilogy: 1901-1980</td><td style="border: 1px solid white;">25980</td><td style="border: 1px solid white;">583</td><td style="border: 1px solid white;">United States</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">9.60</td><td style="border: 1px solid white;">Documentary</td><td style="border: 1px solid white;">Claude Lanzmann</td><td style="border: 1px solid white;">24</td><td style="border: 1px solid white;">1985.00</td><td style="border: 1px solid white;">Shoah</td><td style="border: 1px solid white;">29136</td><td style="border: 1px solid white;">544</td><td style="border: 1px solid white;">France</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">Gibson Gowland, Jean Hersholt, Chester Conklyn</td><td style="border: 1px solid white;">9.60</td><td style="border: 1px solid white;">Drama</td><td style="border: 1px solid white;">Erich Von Stroheim</td><td style="border: 1px solid white;">58</td><td style="border: 1px solid white;">1924.00</td><td style="border: 1px solid white;">Greed</td><td style="border: 1px solid white;">16567</td><td style="border: 1px solid white;">100</td><td style="border: 1px solid white;">United States</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">Tatsuya Nakadai, Akira Ishihama, Shima Iwashita, Akira Ishihama</td><td style="border: 1px solid white;">9.40</td><td style="border: 1px solid white;">Drama</td><td style="border: 1px solid white;">Masaki Kobayashi</td><td style="border: 1px solid white;">68</td><td style="border: 1px solid white;">1962.00</td><td style="border: 1px solid white;">Seppuku</td><td style="border: 1px solid white;">27908</td><td style="border: 1px solid white;">135</td><td style="border: 1px solid white;">Japan</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">Takashi Shimura, Toshiro Mifune, Yoshio Inaba, Seiji Miyaguchi</td><td style="border: 1px solid white;">9.40</td><td style="border: 1px solid white;">Adventure</td><td style="border: 1px solid white;">Akira Kurosawa</td><td style="border: 1px solid white;">325</td><td style="border: 1px solid white;">1954.00</td><td style="border: 1px solid white;">Shichi-nin no Samurai</td><td style="border: 1px solid white;">23395</td><td style="border: 1px solid white;">200</td><td style="border: 1px solid white;">Japan</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[5]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: search, Number of rows: 10, Number of columns: 10</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We can see top productions like The Godfather or Greed. A good idea could be to smooth the avg_vote using a Linear Regression and make it more representative. To create our model we can use the votes, the category, the duration, ... However it could be really usefull to use the director and the main actors.</p>
<p>By using regular expressions we can extract the five main actors for each movie.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[6]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">filmtv_movies</span> <span class="o">=</span> <span class="n">vDataFrame</span><span class="p">(</span><span class="s2">&quot;filmtv_movies&quot;</span><span class="p">)</span>
<span class="n">filmtv_movies</span><span class="p">[</span><span class="s2">&quot;actors&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">fillna</span><span class="p">(</span><span class="s2">&quot;noactors&quot;</span><span class="p">)</span>
<span class="n">filmtv_movies</span><span class="o">.</span><span class="n">regexp</span><span class="p">(</span><span class="n">column</span> <span class="o">=</span> <span class="s2">&quot;actors&quot;</span><span class="p">,</span>
                     <span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;substr&quot;</span><span class="p">,</span>
                     <span class="n">pattern</span> <span class="o">=</span> <span class="s1">&#39;[^,]+&#39;</span><span class="p">,</span>
                     <span class="n">occurrence</span> <span class="o">=</span> <span class="mi">1</span><span class="p">,</span>
                     <span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;actor&quot;</span><span class="p">)</span>

<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="mi">5</span><span class="p">):</span>
    <span class="n">filmtv_movies2</span> <span class="o">=</span> <span class="n">vDataFrame</span><span class="p">(</span><span class="s2">&quot;filmtv_movies&quot;</span><span class="p">)</span>
    <span class="n">filmtv_movies2</span><span class="o">.</span><span class="n">regexp</span><span class="p">(</span><span class="n">column</span> <span class="o">=</span> <span class="s2">&quot;actors&quot;</span><span class="p">,</span>
                          <span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;substr&quot;</span><span class="p">,</span>
                          <span class="n">pattern</span> <span class="o">=</span> <span class="s1">&#39;[^,]+&#39;</span><span class="p">,</span>
                          <span class="n">occurrence</span> <span class="o">=</span> <span class="n">i</span><span class="p">,</span>
                          <span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;actor&quot;</span><span class="p">)</span>
    <span class="n">filmtv_movies</span> <span class="o">=</span> <span class="n">filmtv_movies</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">filmtv_movies2</span><span class="p">)</span>
<span class="n">filmtv_movies</span><span class="p">[</span><span class="s2">&quot;actor&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">describe</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>3025 element(s) was/were filled
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>value</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>name</b></td><td style="border: 1px solid white;">"actor"</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>dtype</b></td><td style="border: 1px solid white;">varchar(2218)</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>unique</b></td><td style="border: 1px solid white;">75086.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>Others</b></td><td style="border: 1px solid white;">191158</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>None</b></td><td style="border: 1px solid white;">18889</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>noactors</b></td><td style="border: 1px solid white;">3025</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>Totò</b></td><td style="border: 1px solid white;">100</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b> Ciccio Ingrassia</b></td><td style="border: 1px solid white;">91</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>John Wayne</b></td><td style="border: 1px solid white;">85</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>Franco Franchi</b></td><td style="border: 1px solid white;">83</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[6]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>By aggregating the data, we can obtain the number of castings and the number of votes by actor. We can normalize the data using the min-max method and obtain an indicator of notoriety.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[7]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">actors_stats</span> <span class="o">=</span> <span class="n">filmtv_movies</span><span class="o">.</span><span class="n">groupby</span><span class="p">(</span><span class="n">columns</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;actor&quot;</span><span class="p">],</span> 
                                     <span class="n">expr</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;SUM(votes) AS notoriety_actors&quot;</span><span class="p">,</span>
                                             <span class="s2">&quot;COUNT(actors) AS castings_actors&quot;</span><span class="p">])</span>
<span class="n">actors_stats</span><span class="o">.</span><span class="n">filter</span><span class="p">(</span><span class="n">expr</span> <span class="o">=</span> <span class="s2">&quot;actor != &#39;noactors&#39;&quot;</span><span class="p">)</span>
<span class="n">actors_stats</span><span class="p">[</span><span class="s2">&quot;notoriety_actors&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">normalize</span><span class="p">(</span><span class="n">method</span> <span class="o">=</span> <span class="s1">&#39;minmax&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>2 element(s) was/were filtered
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>actor</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>notoriety_actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>castings_actors</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;"> Julie Bishop</td><td style="border: 1px solid white;">0.002846396259022</td><td style="border: 1px solid white;">3</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;"> Keith Morris</td><td style="border: 1px solid white;">0.000101657009251</td><td style="border: 1px solid white;">1</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;"> Paul Lukather</td><td style="border: 1px solid white;">0.000508285046254</td><td style="border: 1px solid white;">1</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">Bronson Pinchot</td><td style="border: 1px solid white;">0E-15</td><td style="border: 1px solid white;">1</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;"> Ben Gavin</td><td style="border: 1px solid white;">0.001219884111009</td><td style="border: 1px solid white;">2</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[7]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: groupby, Number of rows: 75085, Number of columns: 3</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Let's look at the top ten actors using our notoriety indicator.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">actors_stats</span><span class="o">.</span><span class="n">search</span><span class="p">(</span><span class="n">order_by</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;notoriety_actors&quot;</span> <span class="p">:</span> <span class="s2">&quot;desc&quot;</span><span class="p">,</span> 
                                <span class="s2">&quot;castings_actors&quot;</span> <span class="p">:</span> <span class="s2">&quot;desc&quot;</span><span class="p">},</span> 
                    <span class="n">limit</span> <span class="o">=</span> <span class="mi">10</span><span class="p">)</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">10</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>actor</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>notoriety_actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>castings_actors</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">Robert De Niro</td><td style="border: 1px solid white;">1.000000000000000</td><td style="border: 1px solid white;">57</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;"> Morgan Freeman</td><td style="border: 1px solid white;">0.861441496391176</td><td style="border: 1px solid white;">52</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">Clint Eastwood</td><td style="border: 1px solid white;">0.856460302937888</td><td style="border: 1px solid white;">43</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">Tom Cruise</td><td style="border: 1px solid white;">0.820372064653858</td><td style="border: 1px solid white;">34</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">Johnny Depp</td><td style="border: 1px solid white;">0.814882586154315</td><td style="border: 1px solid white;">34</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>5</b></td><td style="border: 1px solid white;">Tom Hanks</td><td style="border: 1px solid white;">0.771881671241232</td><td style="border: 1px solid white;">37</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>6</b></td><td style="border: 1px solid white;"> Samuel L. Jackson</td><td style="border: 1px solid white;">0.742502795567754</td><td style="border: 1px solid white;">49</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>7</b></td><td style="border: 1px solid white;">Brad Pitt</td><td style="border: 1px solid white;">0.727762529226390</td><td style="border: 1px solid white;">26</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>8</b></td><td style="border: 1px solid white;">Leonardo DiCaprio</td><td style="border: 1px solid white;">0.715462031107045</td><td style="border: 1px solid white;">20</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>9</b></td><td style="border: 1px solid white;">Al Pacino</td><td style="border: 1px solid white;">0.648368405001525</td><td style="border: 1px solid white;">40</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[8]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: search, Number of rows: 10, Number of columns: 3</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We can see very popular actors like Robert De Niro, Morgan Freeman and Clint Eastwood !</p>
<p>Let's do the same for the movie directors.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[9]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">director_stats</span> <span class="o">=</span> <span class="n">filmtv_movies</span><span class="o">.</span><span class="n">groupby</span><span class="p">(</span><span class="n">columns</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;director&quot;</span><span class="p">],</span> 
                                       <span class="n">expr</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;SUM(votes) AS notoriety_director&quot;</span><span class="p">,</span>
                                               <span class="s2">&quot;COUNT(director) AS castings_director&quot;</span><span class="p">])</span>
<span class="n">director_stats</span><span class="p">[</span><span class="s2">&quot;notoriety_director&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">normalize</span><span class="p">(</span><span class="n">method</span> <span class="o">=</span> <span class="s1">&#39;minmax&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>notoriety_director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>castings_director</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">Crane Wilbur</td><td style="border: 1px solid white;">0.000953206239168</td><td style="border: 1px solid white;">8</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">Julien Hervé, Philippe Mechelen</td><td style="border: 1px solid white;">0.000346620450607</td><td style="border: 1px solid white;">4</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">Michael Lembeck</td><td style="border: 1px solid white;">0.007279029462738</td><td style="border: 1px solid white;">24</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">Burt Kennedy</td><td style="border: 1px solid white;">0.018630849220104</td><td style="border: 1px solid white;">80</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">Gerry Anderson</td><td style="border: 1px solid white;">0.000259965337955</td><td style="border: 1px solid white;">4</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[9]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: groupby, Number of rows: 19161, Number of columns: 3</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Let's now look at the top 10 movie directors.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[10]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">director_stats</span><span class="o">.</span><span class="n">search</span><span class="p">(</span><span class="n">order_by</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;notoriety_director&quot;</span> <span class="p">:</span> <span class="s2">&quot;desc&quot;</span><span class="p">,</span> 
                                  <span class="s2">&quot;castings_director&quot;</span> <span class="p">:</span> <span class="s2">&quot;desc&quot;</span> <span class="p">},</span> 
                      <span class="n">limit</span> <span class="o">=</span> <span class="mi">10</span><span class="p">)</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">10</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>notoriety_director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>castings_director</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">Steven Spielberg</td><td style="border: 1px solid white;">1.000000000000000</td><td style="border: 1px solid white;">132</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">Woody Allen</td><td style="border: 1px solid white;">0.962045060658579</td><td style="border: 1px solid white;">192</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">Clint Eastwood</td><td style="border: 1px solid white;">0.893067590987868</td><td style="border: 1px solid white;">152</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">Martin Scorsese</td><td style="border: 1px solid white;">0.829289428076256</td><td style="border: 1px solid white;">140</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">Alfred Hitchcock</td><td style="border: 1px solid white;">0.753379549393414</td><td style="border: 1px solid white;">208</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>5</b></td><td style="border: 1px solid white;">Quentin Tarantino</td><td style="border: 1px solid white;">0.686741767764298</td><td style="border: 1px solid white;">40</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>6</b></td><td style="border: 1px solid white;">Ridley Scott</td><td style="border: 1px solid white;">0.655459272097054</td><td style="border: 1px solid white;">108</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>7</b></td><td style="border: 1px solid white;">Stanley Kubrick</td><td style="border: 1px solid white;">0.649913344887348</td><td style="border: 1px solid white;">48</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>8</b></td><td style="border: 1px solid white;">Tim Burton</td><td style="border: 1px solid white;">0.588214904679376</td><td style="border: 1px solid white;">72</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>9</b></td><td style="border: 1px solid white;">David Cronenberg</td><td style="border: 1px solid white;">0.513431542461005</td><td style="border: 1px solid white;">84</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[10]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: search, Number of rows: 10, Number of columns: 3</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>It's also working well ! We can see very popular directors like Steven Spielberg, Woody Allen and Clint Eastwood !</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Let's join these new indicators calculated for actors and directors with the main dataset.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[11]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">filmtv_movies_director</span> <span class="o">=</span> <span class="n">filmtv_movies</span><span class="o">.</span><span class="n">join</span><span class="p">(</span>
                                    <span class="n">director_stats</span><span class="p">,</span>
                                    <span class="n">on</span> <span class="o">=</span> <span class="p">{</span><span class="s1">&#39;director&#39;</span><span class="p">:</span> <span class="s1">&#39;director&#39;</span><span class="p">},</span>
                                    <span class="n">how</span> <span class="o">=</span> <span class="s2">&quot;left&quot;</span><span class="p">,</span>
                                    <span class="n">expr1</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;*&quot;</span><span class="p">],</span>
                                    <span class="n">expr2</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;notoriety_director&quot;</span><span class="p">,</span> 
                                             <span class="s2">&quot;castings_director&quot;</span><span class="p">])</span>
<span class="n">filmtv_movies_director_actors</span> <span class="o">=</span> <span class="n">filmtv_movies_director</span><span class="o">.</span><span class="n">join</span><span class="p">(</span>
                                    <span class="n">actors_stats</span><span class="p">,</span>
                                    <span class="n">on</span> <span class="o">=</span> <span class="p">{</span><span class="s1">&#39;actor&#39;</span><span class="p">:</span> <span class="s1">&#39;actor&#39;</span><span class="p">},</span>
                                    <span class="n">how</span> <span class="o">=</span> <span class="s2">&quot;left&quot;</span><span class="p">,</span>
                                    <span class="n">expr1</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;*&quot;</span><span class="p">],</span>
                                    <span class="n">expr2</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;notoriety_actors&quot;</span><span class="p">,</span>
                                             <span class="s2">&quot;castings_actors&quot;</span> <span class="p">])</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We can aggregate the data to get metrics on each movie.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[12]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">filmtv_movies_complete</span> <span class="o">=</span> <span class="n">filmtv_movies_director_actors</span><span class="o">.</span><span class="n">groupby</span><span class="p">(</span>
                                    <span class="n">columns</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;filmtv_id&quot;</span><span class="p">,</span> 
                                               <span class="s2">&quot;title&quot;</span><span class="p">,</span>
                                               <span class="s2">&quot;year&quot;</span><span class="p">,</span>
                                               <span class="s2">&quot;genre&quot;</span><span class="p">,</span>
                                               <span class="s2">&quot;country&quot;</span><span class="p">,</span>
                                               <span class="s2">&quot;avg_vote&quot;</span><span class="p">,</span>
                                               <span class="s2">&quot;votes&quot;</span><span class="p">,</span> 
                                               <span class="s2">&quot;duration&quot;</span><span class="p">,</span> 
                                               <span class="s2">&quot;director&quot;</span><span class="p">,</span> 
                                               <span class="s2">&quot;notoriety_director&quot;</span><span class="p">,</span>
                                               <span class="s2">&quot;castings_director&quot;</span><span class="p">],</span>
                                    <span class="n">expr</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;SUM(notoriety_actors) AS notoriety_actors&quot;</span><span class="p">,</span>
                                            <span class="s2">&quot;SUM(castings_actors) AS castings_actors&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Let's compute some statistics on our dataset.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[13]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">filmtv_movies_complete</span><span class="o">.</span><span class="n">statistics</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>skewness</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>kurtosis</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>count</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>avg</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>stddev</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>min</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>10%</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>25%</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>median</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>75%</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>90%</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>max</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"filmtv_id"</b></td><td style="border: 1px solid white;">1.56007348523374</td><td style="border: 1px solid white;">1.73528867715783</td><td style="border: 1px solid white;">53397.0</td><td style="border: 1px solid white;">44479.2030451149</td><td style="border: 1px solid white;">42397.2726447436</td><td style="border: 1px solid white;">18.0</td><td style="border: 1px solid white;">6129.05882352941</td><td style="border: 1px solid white;">14979.68</td><td style="border: 1px solid white;">31301.28125</td><td style="border: 1px solid white;">56937.2234042553</td><td style="border: 1px solid white;">124569.488372093</td><td style="border: 1px solid white;">179937.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"year"</b></td><td style="border: 1px solid white;">-0.852345359659759</td><td style="border: 1px solid white;">-0.156613762780327</td><td style="border: 1px solid white;">53387.0</td><td style="border: 1px solid white;">1990.97834678854</td><td style="border: 1px solid white;">22.9007675325025</td><td style="border: 1px solid white;">1897.0</td><td style="border: 1px solid white;">1956.0</td><td style="border: 1px solid white;">1975.0</td><td style="border: 1px solid white;">1997.0</td><td style="border: 1px solid white;">2010.0</td><td style="border: 1px solid white;">2015.0</td><td style="border: 1px solid white;">2019.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"avg_vote"</b></td><td style="border: 1px solid white;">-0.37647440486592</td><td style="border: 1px solid white;">0.208912419202391</td><td style="border: 1px solid white;">53397.0</td><td style="border: 1px solid white;">5.84477966926981</td><td style="border: 1px solid white;">1.52365642568413</td><td style="border: 1px solid white;">0.5</td><td style="border: 1px solid white;">4.0</td><td style="border: 1px solid white;">5.0</td><td style="border: 1px solid white;">6.0</td><td style="border: 1px solid white;">7.0</td><td style="border: 1px solid white;">7.8</td><td style="border: 1px solid white;">10.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"votes"</b></td><td style="border: 1px solid white;">5.90414840068652</td><td style="border: 1px solid white;">51.1847237301231</td><td style="border: 1px solid white;">53397.0</td><td style="border: 1px solid white;">24.5417532820196</td><td style="border: 1px solid white;">60.9757267844589</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">2.0</td><td style="border: 1px solid white;">5.0</td><td style="border: 1px solid white;">18.0</td><td style="border: 1px solid white;">63.0</td><td style="border: 1px solid white;">1222.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"duration"</b></td><td style="border: 1px solid white;">65.2361714757649</td><td style="border: 1px solid white;">8825.08538213742</td><td style="border: 1px solid white;">53397.0</td><td style="border: 1px solid white;">98.4667865235875</td><td style="border: 1px solid white;">35.333011278471</td><td style="border: 1px solid white;">40.0</td><td style="border: 1px solid white;">80.0</td><td style="border: 1px solid white;">89.0</td><td style="border: 1px solid white;">95.0</td><td style="border: 1px solid white;">105.0</td><td style="border: 1px solid white;">120.0</td><td style="border: 1px solid white;">5280.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"notoriety_director"</b></td><td style="border: 1px solid white;">6.4227622176949</td><td style="border: 1px solid white;">55.9557539923717</td><td style="border: 1px solid white;">53335.0</td><td style="border: 1px solid white;">0.0302423093689068</td><td style="border: 1px solid white;">0.0811197000890974</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">8.6655112652e-05</td><td style="border: 1px solid white;">0.000606585788562</td><td style="border: 1px solid white;">0.004159445407279</td><td style="border: 1px solid white;">0.021663778162912</td><td style="border: 1px solid white;">0.079896013864818</td><td style="border: 1px solid white;">1.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"castings_director"</b></td><td style="border: 1px solid white;">1.88337156113328</td><td style="border: 1px solid white;">3.88677088572349</td><td style="border: 1px solid white;">53335.0</td><td style="border: 1px solid white;">41.6281241211212</td><td style="border: 1px solid white;">48.0189983882059</td><td style="border: 1px solid white;">4.0</td><td style="border: 1px solid white;">4.0</td><td style="border: 1px solid white;">8.0</td><td style="border: 1px solid white;">24.0</td><td style="border: 1px solid white;">60.0</td><td style="border: 1px solid white;">108.0</td><td style="border: 1px solid white;">292.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"notoriety_actors"</b></td><td style="border: 1px solid white;">2.71318705831567</td><td style="border: 1px solid white;">9.53988861148741</td><td style="border: 1px solid white;">50372.0</td><td style="border: 1px solid white;">0.15134690572669</td><td style="border: 1px solid white;">0.228791818808308</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.001321541120261</td><td style="border: 1px solid white;">0.00955575886957388</td><td style="border: 1px solid white;">0.0577117542254538</td><td style="border: 1px solid white;">0.195750101657009</td><td style="border: 1px solid white;">0.4269318708508</td><td style="border: 1px solid white;">2.231473010064044</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"castings_actors"</b></td><td style="border: 1px solid white;">1.35229605722325</td><td style="border: 1px solid white;">2.37935889949536</td><td style="border: 1px solid white;">50372.0</td><td style="border: 1px solid white;">39.0502660208052</td><td style="border: 1px solid white;">34.0961946541501</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">5.0</td><td style="border: 1px solid white;">11.0</td><td style="border: 1px solid white;">30.0</td><td style="border: 1px solid white;">58.0</td><td style="border: 1px solid white;">86.0</td><td style="border: 1px solid white;">267.0</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[13]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>By looking at the distribution of the 'year', we can create 3 movies categories.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[14]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">filmtv_movies_complete</span><span class="o">.</span><span class="n">case_when</span><span class="p">(</span><span class="s1">&#39;period&#39;</span><span class="p">,</span> 
                                 <span class="p">{</span> <span class="s2">&quot;year &lt; 1990&quot;</span> <span class="p">:</span> <span class="s1">&#39;Old&#39;</span><span class="p">,</span> 
                                  <span class="s2">&quot;year &gt;= 2000&quot;</span> <span class="p">:</span> <span class="s1">&#39;Recent&#39;</span><span class="p">},</span> 
                                  <span class="n">others</span> <span class="o">=</span> <span class="s1">&#39;90s&#39;</span><span class="p">)</span> 
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>filmtv_id</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>title</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>year</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>genre</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>country</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>avg_vote</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>votes</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>duration</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>notoriety_director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>castings_director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>notoriety_actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>castings_actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>period</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">84957</td><td style="border: 1px solid white;">Umi ga kikoeru</td><td style="border: 1px solid white;">1993.00</td><td style="border: 1px solid white;">Animation</td><td style="border: 1px solid white;">Japan</td><td style="border: 1px solid white;">5.80</td><td style="border: 1px solid white;">13</td><td style="border: 1px solid white;">72</td><td style="border: 1px solid white;">Tomomi Mochizuki</td><td style="border: 1px solid white;">0.001039861351820</td><td style="border: 1px solid white;">4</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">90s</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">4583</td><td style="border: 1px solid white;">The Concrete Cowboys</td><td style="border: 1px solid white;">1979.00</td><td style="border: 1px solid white;">Thriller</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">6.00</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">120</td><td style="border: 1px solid white;">Burt Kennedy</td><td style="border: 1px solid white;">0.018630849220104</td><td style="border: 1px solid white;">80</td><td style="border: 1px solid white;">0.025007624275694</td><td style="border: 1px solid white;">21</td><td style="border: 1px solid white;">Old</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">52224</td><td style="border: 1px solid white;">Nero Wolfe - La scatola rossa</td><td style="border: 1px solid white;">2012.00</td><td style="border: 1px solid white;">Thriller</td><td style="border: 1px solid white;">Italy</td><td style="border: 1px solid white;">6.00</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">100</td><td style="border: 1px solid white;">Riccardo Donna</td><td style="border: 1px solid white;">0.009965337954939</td><td style="border: 1px solid white;">56</td><td style="border: 1px solid white;">0.094642675612484</td><td style="border: 1px solid white;">57</td><td style="border: 1px solid white;">Recent</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">51854</td><td style="border: 1px solid white;">La Sirga</td><td style="border: 1px solid white;">2012.00</td><td style="border: 1px solid white;">Drama</td><td style="border: 1px solid white;">Colombia, France, Mexico</td><td style="border: 1px solid white;">6.00</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">88</td><td style="border: 1px solid white;">William Vega</td><td style="border: 1px solid white;">0.000086655112652</td><td style="border: 1px solid white;">4</td><td style="border: 1px solid white;">0.000406628037004</td><td style="border: 1px solid white;">4</td><td style="border: 1px solid white;">Recent</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">8538</td><td style="border: 1px solid white;">The Plague</td><td style="border: 1px solid white;">1992.00</td><td style="border: 1px solid white;">Drama</td><td style="border: 1px solid white;">France, Argentina</td><td style="border: 1px solid white;">5.50</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">135</td><td style="border: 1px solid white;">Luis Puenzo</td><td style="border: 1px solid white;">0.001906412478336</td><td style="border: 1px solid white;">12</td><td style="border: 1px solid white;">0.675714140489987</td><td style="border: 1px solid white;">98</td><td style="border: 1px solid white;">90s</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[14]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: groupby, Number of rows: 53397, Number of columns: 14</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Let's now look at the countries which made the biggest number of movies.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[15]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">filmtv_movies_complete</span><span class="o">.</span><span class="n">groupby</span><span class="p">(</span><span class="n">columns</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;country&quot;</span><span class="p">],</span> 
                               <span class="n">expr</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;COUNT(*)&quot;</span><span class="p">])</span><span class="o">.</span><span class="n">sort</span><span class="p">({</span><span class="s2">&quot;count&quot;</span> <span class="p">:</span> <span class="s2">&quot;desc&quot;</span><span class="p">})</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">10</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>country</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>COUNT</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">21968</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">Italy</td><td style="border: 1px solid white;">9071</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">France</td><td style="border: 1px solid white;">3045</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">Great Britain</td><td style="border: 1px solid white;">2524</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">Germany</td><td style="border: 1px solid white;">1699</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>5</b></td><td style="border: 1px solid white;">Japan</td><td style="border: 1px solid white;">1133</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>6</b></td><td style="border: 1px solid white;">Canada</td><td style="border: 1px solid white;">1053</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>7</b></td><td style="border: 1px solid white;">Spain</td><td style="border: 1px solid white;">563</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>8</b></td><td style="border: 1px solid white;">Italy, France</td><td style="border: 1px solid white;">403</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>9</b></td><td style="border: 1px solid white;">Hong Kong</td><td style="border: 1px solid white;">380</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[15]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: groupby, Number of rows: 2395, Number of columns: 2</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>This variable can be used to create the film main language.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[16]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Language Discretization</span>
<span class="n">Arabic_Middle_Est</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;Arab&quot;</span><span class="p">,</span> <span class="s2">&quot;Iran&quot;</span><span class="p">,</span> <span class="s2">&quot;Turkey&quot;</span><span class="p">,</span> <span class="s2">&quot;Egypt&quot;</span><span class="p">,</span> <span class="s2">&quot;Tunisia&quot;</span><span class="p">,</span>
                     <span class="s2">&quot;Lebanon&quot;</span><span class="p">,</span> <span class="s2">&quot;Palestine&quot;</span><span class="p">,</span> <span class="s2">&quot;Morocco&quot;</span><span class="p">,</span> <span class="s2">&quot;Iraq&quot;</span><span class="p">,</span>
                     <span class="s2">&quot;Sudan&quot;</span><span class="p">,</span> <span class="s2">&quot;Algeria&quot;</span><span class="p">,</span> <span class="s2">&quot;Yemen&quot;</span><span class="p">,</span> <span class="s2">&quot;Afghanistan&quot;</span><span class="p">,</span>
                     <span class="s2">&quot;Azerbaijan&quot;</span><span class="p">,</span> <span class="s2">&quot;Kazakhstan&quot;</span><span class="p">,</span> <span class="s2">&quot;Kyrgyzstan&quot;</span><span class="p">,</span>
                     <span class="s2">&quot;Kurdistan&quot;</span><span class="p">,</span> <span class="s2">&quot;Syria&quot;</span><span class="p">,</span> <span class="s2">&quot;Uzbekistan&quot;</span><span class="p">]</span>
<span class="n">Chinese_Japan_Asian</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;Japan&quot;</span><span class="p">,</span> <span class="s2">&quot;Hong Kong&quot;</span><span class="p">,</span> <span class="s2">&quot;China&quot;</span><span class="p">,</span> <span class="s2">&quot;South Korea&quot;</span><span class="p">,</span> 
                       <span class="s2">&quot;Thailand&quot;</span><span class="p">,</span> <span class="s2">&quot;Philippines&quot;</span><span class="p">,</span> <span class="s2">&quot;Taiwan&quot;</span><span class="p">,</span> <span class="s2">&quot;Indonesia&quot;</span><span class="p">,</span>
                       <span class="s2">&quot;Singapore&quot;</span><span class="p">,</span> <span class="s2">&quot;Malaysia&quot;</span><span class="p">,</span> <span class="s2">&quot;Vietnam&quot;</span><span class="p">,</span> <span class="s2">&quot;Laos&quot;</span><span class="p">,</span> <span class="s2">&quot;Cambodia&quot;</span><span class="p">,</span>
                       <span class="s2">&quot;Bhutan&quot;</span><span class="p">]</span>
<span class="n">Indian</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;India&quot;</span><span class="p">,</span> <span class="s2">&quot;Pakistan&quot;</span><span class="p">,</span> <span class="s2">&quot;Nepal&quot;</span><span class="p">,</span> <span class="s2">&quot;Sri Lanka&quot;</span><span class="p">,</span> <span class="s2">&quot;Bangladesh&quot;</span><span class="p">]</span>
<span class="n">Hebrew</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;Israel&quot;</span><span class="p">]</span>
<span class="n">Spanish_Portuguese</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;Spain&quot;</span><span class="p">,</span> <span class="s2">&quot;Portugal&quot;</span><span class="p">,</span> <span class="s2">&quot;Mexico&quot;</span><span class="p">,</span> <span class="s2">&quot;Brasil&quot;</span><span class="p">,</span> <span class="s2">&quot;Chile&quot;</span><span class="p">,</span>
                      <span class="s2">&quot;Argentina&quot;</span><span class="p">,</span> <span class="s2">&quot;Colombia&quot;</span><span class="p">,</span> <span class="s2">&quot;Cuba&quot;</span><span class="p">,</span> <span class="s2">&quot;Venezuela&quot;</span><span class="p">,</span> <span class="s2">&quot;Peru&quot;</span><span class="p">,</span>
                      <span class="s2">&quot;Uruguay&quot;</span><span class="p">,</span> <span class="s2">&quot;Dominican Republic&quot;</span><span class="p">,</span> <span class="s2">&quot;Ecuador&quot;</span><span class="p">,</span> <span class="s2">&quot;Guatemala&quot;</span><span class="p">,</span>
                      <span class="s2">&quot;Costa Rica&quot;</span><span class="p">,</span> <span class="s2">&quot;Paraguay&quot;</span><span class="p">,</span> <span class="s2">&quot;Bolivia&quot;</span><span class="p">]</span>
<span class="n">English</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;United States&quot;</span><span class="p">,</span> <span class="s2">&quot;England&quot;</span><span class="p">,</span> <span class="s2">&quot;Great Britain&quot;</span><span class="p">,</span> <span class="s2">&quot;Ireland&quot;</span><span class="p">,</span>
           <span class="s2">&quot;Australia&quot;</span><span class="p">,</span> <span class="s2">&quot;New Zealand&quot;</span><span class="p">,</span> <span class="s2">&quot;South Africa&quot;</span><span class="p">]</span>
<span class="n">French</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;France&quot;</span><span class="p">,</span> <span class="s2">&quot;Canada&quot;</span><span class="p">,</span> <span class="s2">&quot;Belgium&quot;</span><span class="p">,</span> <span class="s2">&quot;Switzerland&quot;</span><span class="p">,</span> <span class="s2">&quot;Luxembourg&quot;</span><span class="p">]</span>
<span class="n">Italian</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;Italy&quot;</span><span class="p">]</span>
<span class="n">German_North_Europe</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;German&quot;</span><span class="p">,</span> <span class="s2">&quot;Austria&quot;</span><span class="p">,</span> <span class="s2">&quot;Holland&quot;</span><span class="p">,</span> <span class="s2">&quot;Netherlands&quot;</span><span class="p">,</span> <span class="s2">&quot;Denmark&quot;</span><span class="p">,</span>
                       <span class="s2">&quot;Norway&quot;</span><span class="p">,</span> <span class="s2">&quot;Iceland&quot;</span><span class="p">,</span> <span class="s2">&quot;Finland&quot;</span><span class="p">,</span> <span class="s2">&quot;Sweden&quot;</span><span class="p">,</span> <span class="s2">&quot;Greenland&quot;</span><span class="p">]</span>
<span class="n">Russian_Est_Europe</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;Russia&quot;</span><span class="p">,</span> <span class="s2">&quot;Soviet Union&quot;</span><span class="p">,</span> <span class="s2">&quot;Yugoslavia&quot;</span><span class="p">,</span> <span class="s2">&quot;Czechoslovakia&quot;</span><span class="p">,</span>
                      <span class="s2">&quot;Poland&quot;</span><span class="p">,</span> <span class="s2">&quot;Bulgaria&quot;</span><span class="p">,</span> <span class="s2">&quot;Croatia&quot;</span><span class="p">,</span> <span class="s2">&quot;Czech Republic&quot;</span><span class="p">,</span> <span class="s2">&quot;Serbia&quot;</span><span class="p">,</span>
                      <span class="s2">&quot;Ukraine&quot;</span><span class="p">,</span> <span class="s2">&quot;Slovenia&quot;</span><span class="p">,</span> <span class="s2">&quot;Lithuania&quot;</span><span class="p">,</span> <span class="s2">&quot;Latvia&quot;</span><span class="p">,</span> <span class="s2">&quot;Estonia&quot;</span><span class="p">,</span> 
                      <span class="s2">&quot;Bosnia and Herzegovina&quot;</span><span class="p">,</span> <span class="s2">&quot;Georgia&quot;</span><span class="p">]</span>
<span class="n">Grec_Balkan</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;Greece&quot;</span><span class="p">,</span> <span class="s2">&quot;Macedonia&quot;</span><span class="p">,</span> <span class="s2">&quot;Cyprus&quot;</span><span class="p">,</span> <span class="s2">&quot;Romania&quot;</span><span class="p">,</span> <span class="s2">&quot;Armenia&quot;</span><span class="p">,</span> <span class="s2">&quot;Hungary&quot;</span><span class="p">,</span>
               <span class="s2">&quot;Albania&quot;</span><span class="p">,</span> <span class="s2">&quot;Malta&quot;</span><span class="p">]</span>

<span class="c1"># Creation of the new feature</span>
<span class="n">filmtv_movies_complete</span><span class="o">.</span><span class="n">case_when</span><span class="p">(</span><span class="s1">&#39;language_area&#39;</span><span class="p">,</span> 
        <span class="p">{</span><span class="s2">&quot;REGEXP_LIKE(Country, &#39;</span><span class="si">{}</span><span class="s2">&#39;)&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="s2">&quot;|&quot;</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">Arabic_Middle_Est</span><span class="p">)):</span> <span class="s1">&#39;Arabic_Middle_Est&#39;</span><span class="p">,</span>
         <span class="s2">&quot;REGEXP_LIKE(Country, &#39;</span><span class="si">{}</span><span class="s2">&#39;)&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="s2">&quot;|&quot;</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">Chinese_Japan_Asian</span><span class="p">)):</span> <span class="s1">&#39;Chinese_Japan_Asian&#39;</span><span class="p">,</span> 
         <span class="s2">&quot;REGEXP_LIKE(Country, &#39;</span><span class="si">{}</span><span class="s2">&#39;)&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="s2">&quot;|&quot;</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">Indian</span><span class="p">)):</span> <span class="s1">&#39;Indian&#39;</span><span class="p">,</span> 
         <span class="s2">&quot;REGEXP_LIKE(Country, &#39;</span><span class="si">{}</span><span class="s2">&#39;)&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="s2">&quot;|&quot;</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">Hebrew</span><span class="p">)):</span> <span class="s1">&#39;Hebrew&#39;</span><span class="p">,</span> 
         <span class="s2">&quot;REGEXP_LIKE(Country, &#39;</span><span class="si">{}</span><span class="s2">&#39;)&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="s2">&quot;|&quot;</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">Spanish_Portuguese</span><span class="p">)):</span> <span class="s1">&#39;Spanish_Portuguese&#39;</span><span class="p">,</span> 
         <span class="s2">&quot;REGEXP_LIKE(Country, &#39;</span><span class="si">{}</span><span class="s2">&#39;)&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="s2">&quot;|&quot;</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">English</span><span class="p">)):</span> <span class="s1">&#39;English&#39;</span><span class="p">,</span>
         <span class="s2">&quot;REGEXP_LIKE(Country, &#39;</span><span class="si">{}</span><span class="s2">&#39;)&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="s2">&quot;|&quot;</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">French</span><span class="p">)):</span> <span class="s1">&#39;French&#39;</span><span class="p">,</span>
         <span class="s2">&quot;REGEXP_LIKE(Country, &#39;</span><span class="si">{}</span><span class="s2">&#39;)&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="s2">&quot;|&quot;</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">Italian</span><span class="p">)):</span> <span class="s1">&#39;Italian&#39;</span><span class="p">,</span>
         <span class="s2">&quot;REGEXP_LIKE(Country, &#39;</span><span class="si">{}</span><span class="s2">&#39;)&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="s2">&quot;|&quot;</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">German_North_Europe</span><span class="p">)):</span> <span class="s1">&#39;German_North_Europe&#39;</span><span class="p">,</span>
         <span class="s2">&quot;REGEXP_LIKE(Country, &#39;</span><span class="si">{}</span><span class="s2">&#39;)&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="s2">&quot;|&quot;</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">Russian_Est_Europe</span><span class="p">)):</span> <span class="s1">&#39;Russian_Est_Europe&#39;</span><span class="p">,</span>
         <span class="s2">&quot;REGEXP_LIKE(Country, &#39;</span><span class="si">{}</span><span class="s2">&#39;)&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="s2">&quot;|&quot;</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">Grec_Balkan</span><span class="p">)):</span> <span class="s1">&#39;Grec_Balkan&#39;</span><span class="p">},</span> 
         <span class="n">others</span> <span class="o">=</span> <span class="s1">&#39;Others&#39;</span><span class="p">)</span> 
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>filmtv_id</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>title</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>year</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>genre</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>country</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>avg_vote</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>votes</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>duration</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>notoriety_director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>castings_director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>notoriety_actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>castings_actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>period</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">84957</td><td style="border: 1px solid white;">Umi ga kikoeru</td><td style="border: 1px solid white;">1993.00</td><td style="border: 1px solid white;">Animation</td><td style="border: 1px solid white;">Japan</td><td style="border: 1px solid white;">5.80</td><td style="border: 1px solid white;">13</td><td style="border: 1px solid white;">72</td><td style="border: 1px solid white;">Tomomi Mochizuki</td><td style="border: 1px solid white;">0.001039861351820</td><td style="border: 1px solid white;">4</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">90s</td><td style="border: 1px solid white;">Chinese_Japan_Asian</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">4583</td><td style="border: 1px solid white;">The Concrete Cowboys</td><td style="border: 1px solid white;">1979.00</td><td style="border: 1px solid white;">Thriller</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">6.00</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">120</td><td style="border: 1px solid white;">Burt Kennedy</td><td style="border: 1px solid white;">0.018630849220104</td><td style="border: 1px solid white;">80</td><td style="border: 1px solid white;">0.025007624275694</td><td style="border: 1px solid white;">21</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">English</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">52224</td><td style="border: 1px solid white;">Nero Wolfe - La scatola rossa</td><td style="border: 1px solid white;">2012.00</td><td style="border: 1px solid white;">Thriller</td><td style="border: 1px solid white;">Italy</td><td style="border: 1px solid white;">6.00</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">100</td><td style="border: 1px solid white;">Riccardo Donna</td><td style="border: 1px solid white;">0.009965337954939</td><td style="border: 1px solid white;">56</td><td style="border: 1px solid white;">0.094642675612484</td><td style="border: 1px solid white;">57</td><td style="border: 1px solid white;">Recent</td><td style="border: 1px solid white;">Italian</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">51854</td><td style="border: 1px solid white;">La Sirga</td><td style="border: 1px solid white;">2012.00</td><td style="border: 1px solid white;">Drama</td><td style="border: 1px solid white;">Colombia, France, Mexico</td><td style="border: 1px solid white;">6.00</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">88</td><td style="border: 1px solid white;">William Vega</td><td style="border: 1px solid white;">0.000086655112652</td><td style="border: 1px solid white;">4</td><td style="border: 1px solid white;">0.000406628037004</td><td style="border: 1px solid white;">4</td><td style="border: 1px solid white;">Recent</td><td style="border: 1px solid white;">Spanish_Portuguese</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">8538</td><td style="border: 1px solid white;">The Plague</td><td style="border: 1px solid white;">1992.00</td><td style="border: 1px solid white;">Drama</td><td style="border: 1px solid white;">France, Argentina</td><td style="border: 1px solid white;">5.50</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">135</td><td style="border: 1px solid white;">Luis Puenzo</td><td style="border: 1px solid white;">0.001906412478336</td><td style="border: 1px solid white;">12</td><td style="border: 1px solid white;">0.675714140489987</td><td style="border: 1px solid white;">98</td><td style="border: 1px solid white;">90s</td><td style="border: 1px solid white;">Spanish_Portuguese</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[16]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: groupby, Number of rows: 53397, Number of columns: 15</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We can do the same for the genres.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[17]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">filmtv_movies_complete</span><span class="o">.</span><span class="n">case_when</span><span class="p">(</span><span class="s1">&#39;Category&#39;</span><span class="p">,</span> 
                     <span class="p">{</span><span class="s2">&quot;REGEXP_LIKE(Genre, &#39;Drama|Noir&#39;)&quot;</span><span class="p">:</span> <span class="s1">&#39;Drama&#39;</span><span class="p">,</span> 
                      <span class="s2">&quot;REGEXP_LIKE(Genre, &#39;Comedy|Grotesque&#39;)&quot;</span><span class="p">:</span> <span class="s1">&#39;Comedy&#39;</span><span class="p">,</span> 
                      <span class="s2">&quot;REGEXP_LIKE(Genre, &#39;Fantasy|Super-hero&#39;)&quot;</span><span class="p">:</span> <span class="s1">&#39;Fantasy&#39;</span><span class="p">,</span> 
                      <span class="s2">&quot;REGEXP_LIKE(Genre, &#39;Romantic|Sperimental|Mélo&#39;)&quot;</span><span class="p">:</span> <span class="s1">&#39;Romantic&#39;</span><span class="p">,</span> 
                      <span class="s2">&quot;REGEXP_LIKE(Genre, &#39;Thriller|Crime|Gangster&#39;)&quot;</span><span class="p">:</span> <span class="s1">&#39;Thriller&#39;</span><span class="p">,</span> 
                      <span class="s2">&quot;REGEXP_LIKE(Genre, &#39;Action|Western|War|Spy&#39;)&quot;</span><span class="p">:</span> <span class="s1">&#39;Action&#39;</span><span class="p">,</span> 
                      <span class="s2">&quot;REGEXP_LIKE(Genre, &#39;Adventure&#39;)&quot;</span><span class="p">:</span> <span class="s1">&#39;Adventure&#39;</span><span class="p">,</span> 
                      <span class="s2">&quot;REGEXP_LIKE(Genre, &#39;Animation&#39;)&quot;</span><span class="p">:</span> <span class="s1">&#39;Animation&#39;</span><span class="p">,</span> 
                      <span class="s2">&quot;REGEXP_LIKE(Genre, &#39;Horror&#39;)&quot;</span><span class="p">:</span> <span class="s1">&#39;Horror&#39;</span><span class="p">},</span> 
                      <span class="n">others</span> <span class="o">=</span> <span class="s1">&#39;Others&#39;</span><span class="p">)</span> 
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>filmtv_id</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>title</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>year</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>genre</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>country</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>avg_vote</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>votes</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>duration</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>notoriety_director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>castings_director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>notoriety_actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>castings_actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>period</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">84957</td><td style="border: 1px solid white;">Umi ga kikoeru</td><td style="border: 1px solid white;">1993.00</td><td style="border: 1px solid white;">Animation</td><td style="border: 1px solid white;">Japan</td><td style="border: 1px solid white;">5.80</td><td style="border: 1px solid white;">13</td><td style="border: 1px solid white;">72</td><td style="border: 1px solid white;">Tomomi Mochizuki</td><td style="border: 1px solid white;">0.001039861351820</td><td style="border: 1px solid white;">4</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">90s</td><td style="border: 1px solid white;">Chinese_Japan_Asian</td><td style="border: 1px solid white;">Animation</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">4583</td><td style="border: 1px solid white;">The Concrete Cowboys</td><td style="border: 1px solid white;">1979.00</td><td style="border: 1px solid white;">Thriller</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">6.00</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">120</td><td style="border: 1px solid white;">Burt Kennedy</td><td style="border: 1px solid white;">0.018630849220104</td><td style="border: 1px solid white;">80</td><td style="border: 1px solid white;">0.025007624275694</td><td style="border: 1px solid white;">21</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">English</td><td style="border: 1px solid white;">Thriller</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">52224</td><td style="border: 1px solid white;">Nero Wolfe - La scatola rossa</td><td style="border: 1px solid white;">2012.00</td><td style="border: 1px solid white;">Thriller</td><td style="border: 1px solid white;">Italy</td><td style="border: 1px solid white;">6.00</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">100</td><td style="border: 1px solid white;">Riccardo Donna</td><td style="border: 1px solid white;">0.009965337954939</td><td style="border: 1px solid white;">56</td><td style="border: 1px solid white;">0.094642675612484</td><td style="border: 1px solid white;">57</td><td style="border: 1px solid white;">Recent</td><td style="border: 1px solid white;">Italian</td><td style="border: 1px solid white;">Thriller</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">51854</td><td style="border: 1px solid white;">La Sirga</td><td style="border: 1px solid white;">2012.00</td><td style="border: 1px solid white;">Drama</td><td style="border: 1px solid white;">Colombia, France, Mexico</td><td style="border: 1px solid white;">6.00</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">88</td><td style="border: 1px solid white;">William Vega</td><td style="border: 1px solid white;">0.000086655112652</td><td style="border: 1px solid white;">4</td><td style="border: 1px solid white;">0.000406628037004</td><td style="border: 1px solid white;">4</td><td style="border: 1px solid white;">Recent</td><td style="border: 1px solid white;">Spanish_Portuguese</td><td style="border: 1px solid white;">Drama</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">8538</td><td style="border: 1px solid white;">The Plague</td><td style="border: 1px solid white;">1992.00</td><td style="border: 1px solid white;">Drama</td><td style="border: 1px solid white;">France, Argentina</td><td style="border: 1px solid white;">5.50</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">135</td><td style="border: 1px solid white;">Luis Puenzo</td><td style="border: 1px solid white;">0.001906412478336</td><td style="border: 1px solid white;">12</td><td style="border: 1px solid white;">0.675714140489987</td><td style="border: 1px solid white;">98</td><td style="border: 1px solid white;">90s</td><td style="border: 1px solid white;">Spanish_Portuguese</td><td style="border: 1px solid white;">Drama</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[17]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: groupby, Number of rows: 53397, Number of columns: 16</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>As the new feature 'Category' is more relevant, we can drop the feature 'genre'.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[18]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">filmtv_movies_complete</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="n">columns</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;genre&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>filmtv_id</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>title</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>year</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>country</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>avg_vote</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>votes</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>duration</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>notoriety_director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>castings_director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>notoriety_actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>castings_actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>period</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">84957</td><td style="border: 1px solid white;">Umi ga kikoeru</td><td style="border: 1px solid white;">1993.00</td><td style="border: 1px solid white;">Japan</td><td style="border: 1px solid white;">5.80</td><td style="border: 1px solid white;">13</td><td style="border: 1px solid white;">72</td><td style="border: 1px solid white;">Tomomi Mochizuki</td><td style="border: 1px solid white;">0.001039861351820</td><td style="border: 1px solid white;">4</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">90s</td><td style="border: 1px solid white;">Chinese_Japan_Asian</td><td style="border: 1px solid white;">Animation</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">4583</td><td style="border: 1px solid white;">The Concrete Cowboys</td><td style="border: 1px solid white;">1979.00</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">6.00</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">120</td><td style="border: 1px solid white;">Burt Kennedy</td><td style="border: 1px solid white;">0.018630849220104</td><td style="border: 1px solid white;">80</td><td style="border: 1px solid white;">0.025007624275694</td><td style="border: 1px solid white;">21</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">English</td><td style="border: 1px solid white;">Thriller</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">52224</td><td style="border: 1px solid white;">Nero Wolfe - La scatola rossa</td><td style="border: 1px solid white;">2012.00</td><td style="border: 1px solid white;">Italy</td><td style="border: 1px solid white;">6.00</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">100</td><td style="border: 1px solid white;">Riccardo Donna</td><td style="border: 1px solid white;">0.009965337954939</td><td style="border: 1px solid white;">56</td><td style="border: 1px solid white;">0.094642675612484</td><td style="border: 1px solid white;">57</td><td style="border: 1px solid white;">Recent</td><td style="border: 1px solid white;">Italian</td><td style="border: 1px solid white;">Thriller</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">51854</td><td style="border: 1px solid white;">La Sirga</td><td style="border: 1px solid white;">2012.00</td><td style="border: 1px solid white;">Colombia, France, Mexico</td><td style="border: 1px solid white;">6.00</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">88</td><td style="border: 1px solid white;">William Vega</td><td style="border: 1px solid white;">0.000086655112652</td><td style="border: 1px solid white;">4</td><td style="border: 1px solid white;">0.000406628037004</td><td style="border: 1px solid white;">4</td><td style="border: 1px solid white;">Recent</td><td style="border: 1px solid white;">Spanish_Portuguese</td><td style="border: 1px solid white;">Drama</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">8538</td><td style="border: 1px solid white;">The Plague</td><td style="border: 1px solid white;">1992.00</td><td style="border: 1px solid white;">France, Argentina</td><td style="border: 1px solid white;">5.50</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">135</td><td style="border: 1px solid white;">Luis Puenzo</td><td style="border: 1px solid white;">0.001906412478336</td><td style="border: 1px solid white;">12</td><td style="border: 1px solid white;">0.675714140489987</td><td style="border: 1px solid white;">98</td><td style="border: 1px solid white;">90s</td><td style="border: 1px solid white;">Spanish_Portuguese</td><td style="border: 1px solid white;">Drama</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[18]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: groupby, Number of rows: 53397, Number of columns: 15</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Let's now look at the missing values.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[19]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">filmtv_movies_complete</span><span class="o">.</span><span class="n">count</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>count</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>percent</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"filmtv_id"</b></td><td style="border: 1px solid white;">53397.0</td><td style="border: 1px solid white;">100.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"avg_vote"</b></td><td style="border: 1px solid white;">53397.0</td><td style="border: 1px solid white;">100.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"votes"</b></td><td style="border: 1px solid white;">53397.0</td><td style="border: 1px solid white;">100.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"duration"</b></td><td style="border: 1px solid white;">53397.0</td><td style="border: 1px solid white;">100.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"period"</b></td><td style="border: 1px solid white;">53397.0</td><td style="border: 1px solid white;">100.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"language_area"</b></td><td style="border: 1px solid white;">53397.0</td><td style="border: 1px solid white;">100.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"Category"</b></td><td style="border: 1px solid white;">53397.0</td><td style="border: 1px solid white;">100.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"title"</b></td><td style="border: 1px solid white;">53395.0</td><td style="border: 1px solid white;">99.996</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"year"</b></td><td style="border: 1px solid white;">53387.0</td><td style="border: 1px solid white;">99.981</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"country"</b></td><td style="border: 1px solid white;">53346.0</td><td style="border: 1px solid white;">99.904</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"director"</b></td><td style="border: 1px solid white;">53335.0</td><td style="border: 1px solid white;">99.884</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"notoriety_director"</b></td><td style="border: 1px solid white;">53335.0</td><td style="border: 1px solid white;">99.884</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"castings_director"</b></td><td style="border: 1px solid white;">53335.0</td><td style="border: 1px solid white;">99.884</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"notoriety_actors"</b></td><td style="border: 1px solid white;">50372.0</td><td style="border: 1px solid white;">94.335</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"castings_actors"</b></td><td style="border: 1px solid white;">50372.0</td><td style="border: 1px solid white;">94.335</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[19]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Let's impute the missing values for 'notoriety_actors' and 'castings_actors' using different techniques. We can then drop the few remaining missing values.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[20]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">filmtv_movies_complete</span><span class="p">[</span><span class="s2">&quot;notoriety_actors&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">fillna</span><span class="p">(</span><span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;median&quot;</span><span class="p">,</span>
                                                  <span class="n">by</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;director&quot;</span><span class="p">,</span>
                                                        <span class="s2">&quot;Category&quot;</span><span class="p">])</span>
<span class="n">filmtv_movies_complete</span><span class="p">[</span><span class="s2">&quot;castings_actors&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">fillna</span><span class="p">(</span><span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;median&quot;</span><span class="p">,</span>
                                                 <span class="n">by</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;director&quot;</span><span class="p">,</span>
                                                       <span class="s2">&quot;Category&quot;</span><span class="p">])</span>
<span class="n">filmtv_movies_complete</span><span class="o">.</span><span class="n">dropna</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>640 element(s) was/were filled
640 element(s) was/were filled
2499 element(s) was/were dropped
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>filmtv_id</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>title</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>year</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>country</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>avg_vote</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>votes</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>duration</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>notoriety_director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>castings_director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>notoriety_actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>castings_actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>period</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">80684</td><td style="border: 1px solid white;">Lake Placid vs. Anaconda</td><td style="border: 1px solid white;">2015.00</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">4.00</td><td style="border: 1px solid white;">5</td><td style="border: 1px solid white;">86</td><td style="border: 1px solid white;">A.B. Stone</td><td style="border: 1px solid white;">0.000346620450607</td><td style="border: 1px solid white;">4</td><td style="border: 1px solid white;">0.016163464470875</td><td style="border: 1px solid white;">21.0</td><td style="border: 1px solid white;">Recent</td><td style="border: 1px solid white;">English</td><td style="border: 1px solid white;">Action</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">56292</td><td style="border: 1px solid white;">Thuppakki</td><td style="border: 1px solid white;">2012.00</td><td style="border: 1px solid white;">India</td><td style="border: 1px solid white;">8.00</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">150</td><td style="border: 1px solid white;">A.R. Murugadoss</td><td style="border: 1px solid white;">0.000086655112652</td><td style="border: 1px solid white;">8</td><td style="border: 1px solid white;">0.000101657009251</td><td style="border: 1px solid white;">5.0</td><td style="border: 1px solid white;">Recent</td><td style="border: 1px solid white;">Indian</td><td style="border: 1px solid white;">Action</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">63738</td><td style="border: 1px solid white;">Ghajini</td><td style="border: 1px solid white;">2008.00</td><td style="border: 1px solid white;">India</td><td style="border: 1px solid white;">8.00</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">183</td><td style="border: 1px solid white;">A.R. Murugadoss</td><td style="border: 1px solid white;">0.000086655112652</td><td style="border: 1px solid white;">8</td><td style="border: 1px solid white;">0.002338111212768</td><td style="border: 1px solid white;">9.0</td><td style="border: 1px solid white;">Recent</td><td style="border: 1px solid white;">Indian</td><td style="border: 1px solid white;">Action</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">49021</td><td style="border: 1px solid white;">Catch .44</td><td style="border: 1px solid white;">2011.00</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">3.90</td><td style="border: 1px solid white;">39</td><td style="border: 1px solid white;">108</td><td style="border: 1px solid white;">Aaron Harvey</td><td style="border: 1px solid white;">0.003292894280763</td><td style="border: 1px solid white;">4</td><td style="border: 1px solid white;">0.885025922537358</td><td style="border: 1px solid white;">95.0</td><td style="border: 1px solid white;">Recent</td><td style="border: 1px solid white;">English</td><td style="border: 1px solid white;">Action</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">17196</td><td style="border: 1px solid white;">City Limits</td><td style="border: 1px solid white;">1985.00</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">4.00</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">90</td><td style="border: 1px solid white;">Aaron Lipstadt</td><td style="border: 1px solid white;">0.001559792027730</td><td style="border: 1px solid white;">16</td><td style="border: 1px solid white;">0.196502998881773</td><td style="border: 1px solid white;">55.0</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">English</td><td style="border: 1px solid white;">Action</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[20]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: groupby, Number of rows: 50898, Number of columns: 15</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Last step before exporting the data is to normalize the numerical columns and to get the dummies of the different categories.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[21]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">filmtv_movies_complete</span><span class="o">.</span><span class="n">normalize</span><span class="p">(</span><span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;minmax&quot;</span><span class="p">,</span>
                                 <span class="n">columns</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;votes&#39;</span><span class="p">,</span> 
                                            <span class="s1">&#39;duration&#39;</span><span class="p">,</span> 
                                            <span class="s1">&#39;notoriety_director&#39;</span><span class="p">,</span>
                                            <span class="s1">&#39;castings_director&#39;</span><span class="p">,</span>
                                            <span class="s1">&#39;notoriety_actors&#39;</span><span class="p">,</span>
                                            <span class="s1">&#39;castings_actors&#39;</span><span class="p">])</span>
<span class="k">for</span> <span class="n">elem</span> <span class="ow">in</span> <span class="p">[</span><span class="s1">&#39;category&#39;</span><span class="p">,</span> <span class="s1">&#39;period&#39;</span><span class="p">,</span> <span class="s1">&#39;language_area&#39;</span><span class="p">]:</span>
    <span class="n">filmtv_movies_complete</span><span class="p">[</span><span class="n">elem</span><span class="p">]</span><span class="o">.</span><span class="n">get_dummies</span><span class="p">(</span><span class="n">drop_first</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We can export the result to the Vertica DataBase.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[22]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">drop_table</span><span class="p">(</span><span class="s2">&quot;filmtv_movies_complete&quot;</span><span class="p">)</span>
<span class="n">filmtv_movies_complete</span><span class="o">.</span><span class="n">to_db</span><span class="p">(</span><span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;filmtv_movies_complete&quot;</span><span class="p">,</span>
                             <span class="n">relation_type</span> <span class="o">=</span> <span class="s2">&quot;table&quot;</span><span class="p">,</span>
                             <span class="n">inplace</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
<span class="n">drop_view</span><span class="p">(</span><span class="s2">&quot;filmtv_movies_mco&quot;</span><span class="p">)</span>
<span class="n">filmtv_movies_complete</span><span class="o">.</span><span class="n">to_db</span><span class="p">(</span><span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;filmtv_movies_mco&quot;</span><span class="p">,</span>
                             <span class="n">relation_type</span> <span class="o">=</span> <span class="s2">&quot;view&quot;</span><span class="p">,</span>
                             <span class="n">db_filter</span> <span class="o">=</span> <span class="s2">&quot;votes &gt; 0.02&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>The table filmtv_movies_complete was successfully dropped.
The view filmtv_movies_mco was successfully dropped.
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_French</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Animation</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>avg_vote</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Romantic</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>period_90s</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>period</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Adventure</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Russian_Est_Europe</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Arabic_Middle_Est</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_German_North_Europe</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Chinese_Japan_Asian</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Italian</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>title</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>notoriety_actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>period_Old</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Fantasy</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>notoriety_director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>filmtv_id</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>votes</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Grec_Balkan</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Horror</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>country</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Others</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Drama</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Indian</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>castings_actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>castings_director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>duration</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>year</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Others</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_English</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Action</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Comedy</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Hebrew</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">4.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Recent</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">A.B. Stone</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">English</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Lake Placid vs. Anaconda</td><td style="border: 1px solid white;">0.0072434057673909</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00034662045060700</td><td style="border: 1px solid white;">80684</td><td style="border: 1px solid white;">0.003276003276003</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.075187969924812</td><td style="border: 1px solid white;">0E-15</td><td style="border: 1px solid white;">0.008778625954198</td><td style="border: 1px solid white;">2015.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Action</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">8.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Recent</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">A.R. Murugadoss</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Indian</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Thuppakki</td><td style="border: 1px solid white;">4.55560111157619e-05</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00008665511265200</td><td style="border: 1px solid white;">56292</td><td style="border: 1px solid white;">0E-15</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">India</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0.0150375939849624</td><td style="border: 1px solid white;">0.013888888888889</td><td style="border: 1px solid white;">0.020992366412214</td><td style="border: 1px solid white;">2012.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Action</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">8.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Recent</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">A.R. Murugadoss</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Indian</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Ghajini</td><td style="border: 1px solid white;">0.00104778825566028</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00008665511265200</td><td style="border: 1px solid white;">63738</td><td style="border: 1px solid white;">0E-15</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">India</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0.0300751879699248</td><td style="border: 1px solid white;">0.013888888888889</td><td style="border: 1px solid white;">0.027290076335878</td><td style="border: 1px solid white;">2008.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Action</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">3.90</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Recent</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Aaron Harvey</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">English</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Catch .44</td><td style="border: 1px solid white;">0.396610632772995</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00329289428076300</td><td style="border: 1px solid white;">49021</td><td style="border: 1px solid white;">0.031122031122031</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.353383458646617</td><td style="border: 1px solid white;">0E-15</td><td style="border: 1px solid white;">0.012977099236641</td><td style="border: 1px solid white;">2011.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Action</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">4.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Aaron Lipstadt</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">English</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">City Limits</td><td style="border: 1px solid white;">0.0880597694865839</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00155979202773000</td><td style="border: 1px solid white;">17196</td><td style="border: 1px solid white;">0E-15</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.203007518796992</td><td style="border: 1px solid white;">0.041666666666667</td><td style="border: 1px solid white;">0.009541984732824</td><td style="border: 1px solid white;">1985.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Action</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[22]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: filmtv_movies_complete, Number of rows: 50898, Number of columns: 37</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Machine-Learning-:-Adjusting-the-Films-Rates">Machine Learning : Adjusting the Films Rates<a class="anchor-link" href="#Machine-Learning-:-Adjusting-the-Films-Rates">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Let's create a model to evaluate an unbiased rate of the different movies.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[23]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.linear_model</span> <span class="k">import</span> <span class="n">ElasticNet</span>
<span class="n">predictors</span> <span class="o">=</span> <span class="n">filmtv_movies_complete</span><span class="o">.</span><span class="n">get_columns</span><span class="p">(</span><span class="n">exclude_columns</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;avg_vote&quot;</span><span class="p">,</span>
                                                                   <span class="s2">&quot;period&quot;</span><span class="p">,</span>
                                                                   <span class="s2">&quot;director&quot;</span><span class="p">,</span>
                                                                   <span class="s2">&quot;language_area&quot;</span><span class="p">,</span>
                                                                   <span class="s2">&quot;title&quot;</span><span class="p">,</span> 
                                                                   <span class="s2">&quot;year&quot;</span><span class="p">,</span>
                                                                   <span class="s2">&quot;country&quot;</span><span class="p">,</span>
                                                                   <span class="s2">&quot;Category&quot;</span><span class="p">])</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">ElasticNet</span><span class="p">(</span><span class="s2">&quot;filmtv_movies_enet&quot;</span><span class="p">,</span> 
                   <span class="n">penalty</span> <span class="o">=</span> <span class="s1">&#39;L2&#39;</span><span class="p">,</span> 
                   <span class="n">max_iter</span> <span class="o">=</span> <span class="mi">1000</span><span class="p">,</span> 
                   <span class="n">solver</span> <span class="o">=</span> <span class="s1">&#39;BFGS&#39;</span><span class="p">)</span>
<span class="n">model</span><span class="o">.</span><span class="n">drop</span><span class="p">()</span>
<span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="s2">&quot;filmtv_movies_mco&quot;</span><span class="p">,</span> <span class="n">predictors</span><span class="p">,</span> <span class="s2">&quot;avg_vote&quot;</span><span class="p">)</span>
<span class="n">model</span><span class="o">.</span><span class="n">regression_report</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>value</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>explained_variance</b></td><td style="border: 1px solid white;">0.462404169262795</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>max_error</b></td><td style="border: 1px solid white;">4.92511236870251</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>median_absolute_error</b></td><td style="border: 1px solid white;">0.593914006552813</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>mean_absolute_error</b></td><td style="border: 1px solid white;">0.717283090338238</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>mean_squared_error</b></td><td style="border: 1px solid white;">0.83495209314207</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>r2</b></td><td style="border: 1px solid white;">0.462391185379635</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[23]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The model is good, we can add it in our vDataFrame.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[24]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">filmtv_movies_complete</span><span class="p">,</span>
              <span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;unbiased_vote&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_French</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Animation</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>avg_vote</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Romantic</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>period_90s</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>period</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Adventure</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Russian_Est_Europe</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Arabic_Middle_Est</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_German_North_Europe</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Chinese_Japan_Asian</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Italian</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>title</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>notoriety_actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>period_Old</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Fantasy</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>notoriety_director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>filmtv_id</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>votes</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Grec_Balkan</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Horror</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>country</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Others</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Drama</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Indian</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>castings_actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>castings_director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>duration</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>year</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Others</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_English</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Action</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Comedy</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Hebrew</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>unbiased_vote</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">4.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Recent</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">A.B. Stone</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">English</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Lake Placid vs. Anaconda</td><td style="border: 1px solid white;">0.0072434057673909</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00034662045060700</td><td style="border: 1px solid white;">80684</td><td style="border: 1px solid white;">0.003276003276003</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.075187969924812</td><td style="border: 1px solid white;">0E-15</td><td style="border: 1px solid white;">0.008778625954198</td><td style="border: 1px solid white;">2015.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Action</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">5.32603426349589</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">8.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Recent</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">A.R. Murugadoss</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Indian</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Thuppakki</td><td style="border: 1px solid white;">4.55560111157619e-05</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00008665511265200</td><td style="border: 1px solid white;">56292</td><td style="border: 1px solid white;">0E-15</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">India</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0.0150375939849624</td><td style="border: 1px solid white;">0.013888888888889</td><td style="border: 1px solid white;">0.020992366412214</td><td style="border: 1px solid white;">2012.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Action</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">5.66634526656623</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">8.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Recent</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">A.R. Murugadoss</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Indian</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Ghajini</td><td style="border: 1px solid white;">0.00104778825566028</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00008665511265200</td><td style="border: 1px solid white;">63738</td><td style="border: 1px solid white;">0E-15</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">India</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0.0300751879699248</td><td style="border: 1px solid white;">0.013888888888889</td><td style="border: 1px solid white;">0.027290076335878</td><td style="border: 1px solid white;">2008.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Action</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">5.68847396790354</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">3.90</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Recent</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Aaron Harvey</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">English</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Catch .44</td><td style="border: 1px solid white;">0.396610632772995</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00329289428076300</td><td style="border: 1px solid white;">49021</td><td style="border: 1px solid white;">0.031122031122031</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.353383458646617</td><td style="border: 1px solid white;">0E-15</td><td style="border: 1px solid white;">0.012977099236641</td><td style="border: 1px solid white;">2011.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Action</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">5.07780968922694</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">4.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Aaron Lipstadt</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">English</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">City Limits</td><td style="border: 1px solid white;">0.0880597694865839</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00155979202773000</td><td style="border: 1px solid white;">17196</td><td style="border: 1px solid white;">0E-15</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.203007518796992</td><td style="border: 1px solid white;">0.041666666666667</td><td style="border: 1px solid white;">0.009541984732824</td><td style="border: 1px solid white;">1985.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Action</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">6.4543100859855</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[24]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: filmtv_movies_complete, Number of rows: 50898, Number of columns: 38</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>As a rate can not be greater than 10 or lesser than 0. We need to adjust the 'unbiased_vote'.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[25]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">filmtv_movies_complete</span><span class="p">[</span><span class="s2">&quot;unbiased_vote&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span>
    <span class="s2">&quot;CASE WHEN </span><span class="si">{}</span><span class="s2"> &gt; 10 THEN 10 WHEN </span><span class="si">{}</span><span class="s2"> &lt; 0 THEN 0 ELSE </span><span class="si">{}</span><span class="s2"> END&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_French</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Animation</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>avg_vote</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Romantic</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>period_90s</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>period</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Adventure</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Russian_Est_Europe</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Arabic_Middle_Est</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_German_North_Europe</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Chinese_Japan_Asian</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Italian</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>title</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>notoriety_actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>period_Old</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Fantasy</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>notoriety_director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>filmtv_id</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>votes</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Grec_Balkan</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Horror</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>country</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Others</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Drama</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Indian</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>castings_actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>castings_director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>duration</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>year</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Others</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_English</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Action</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Comedy</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Hebrew</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>unbiased_vote</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">4.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Recent</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">A.B. Stone</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">English</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Lake Placid vs. Anaconda</td><td style="border: 1px solid white;">0.0072434057673909</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00034662045060700</td><td style="border: 1px solid white;">80684</td><td style="border: 1px solid white;">0.003276003276003</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.075187969924812</td><td style="border: 1px solid white;">0E-15</td><td style="border: 1px solid white;">0.008778625954198</td><td style="border: 1px solid white;">2015.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Action</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">5.32603426349589</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">8.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Recent</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">A.R. Murugadoss</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Indian</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Thuppakki</td><td style="border: 1px solid white;">4.55560111157619e-05</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00008665511265200</td><td style="border: 1px solid white;">56292</td><td style="border: 1px solid white;">0E-15</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">India</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0.0150375939849624</td><td style="border: 1px solid white;">0.013888888888889</td><td style="border: 1px solid white;">0.020992366412214</td><td style="border: 1px solid white;">2012.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Action</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">5.66634526656623</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">8.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Recent</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">A.R. Murugadoss</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Indian</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Ghajini</td><td style="border: 1px solid white;">0.00104778825566028</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00008665511265200</td><td style="border: 1px solid white;">63738</td><td style="border: 1px solid white;">0E-15</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">India</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0.0300751879699248</td><td style="border: 1px solid white;">0.013888888888889</td><td style="border: 1px solid white;">0.027290076335878</td><td style="border: 1px solid white;">2008.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Action</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">5.68847396790354</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">3.90</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Recent</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Aaron Harvey</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">English</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Catch .44</td><td style="border: 1px solid white;">0.396610632772995</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00329289428076300</td><td style="border: 1px solid white;">49021</td><td style="border: 1px solid white;">0.031122031122031</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.353383458646617</td><td style="border: 1px solid white;">0E-15</td><td style="border: 1px solid white;">0.012977099236641</td><td style="border: 1px solid white;">2011.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Action</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">5.07780968922694</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">4.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Aaron Lipstadt</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">English</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">City Limits</td><td style="border: 1px solid white;">0.0880597694865839</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00155979202773000</td><td style="border: 1px solid white;">17196</td><td style="border: 1px solid white;">0E-15</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.203007518796992</td><td style="border: 1px solid white;">0.041666666666667</td><td style="border: 1px solid white;">0.009541984732824</td><td style="border: 1px solid white;">1985.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Action</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">6.4543100859855</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[25]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: filmtv_movies_complete, Number of rows: 50898, Number of columns: 38</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We can then look at the new top movies.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[26]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">filmtv_movies_complete</span><span class="o">.</span><span class="n">search</span><span class="p">(</span><span class="n">usecols</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;filmtv_id&#39;</span><span class="p">,</span>
                                         <span class="s1">&#39;title&#39;</span><span class="p">,</span>
                                         <span class="s1">&#39;year&#39;</span><span class="p">,</span>
                                         <span class="s1">&#39;country&#39;</span><span class="p">,</span>
                                         <span class="s1">&#39;avg_vote&#39;</span><span class="p">,</span>
                                         <span class="s1">&#39;unbiased_vote&#39;</span><span class="p">,</span>
                                         <span class="s1">&#39;votes&#39;</span><span class="p">,</span>
                                         <span class="s1">&#39;duration&#39;</span><span class="p">,</span>
                                         <span class="s1">&#39;director&#39;</span><span class="p">,</span>
                                         <span class="s1">&#39;notoriety_director&#39;</span><span class="p">,</span>
                                         <span class="s1">&#39;castings_director&#39;</span><span class="p">,</span>
                                         <span class="s1">&#39;notoriety_actors&#39;</span><span class="p">,</span>
                                         <span class="s1">&#39;castings_actors&#39;</span><span class="p">,</span>
                                         <span class="s1">&#39;period&#39;</span><span class="p">,</span>
                                         <span class="s1">&#39;language_area&#39;</span><span class="p">],</span>
                              <span class="n">order_by</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;unbiased_vote&quot;</span> <span class="p">:</span> <span class="s2">&quot;desc&quot;</span><span class="p">,</span> 
                                          <span class="s2">&quot;avg_vote&quot;</span> <span class="p">:</span> <span class="s2">&quot;desc&quot;</span><span class="p">})</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">10</span><span class="p">)</span> 
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>filmtv_id</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>title</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>year</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>country</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>avg_vote</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>unbiased_vote</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>votes</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>duration</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>notoriety_director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>castings_director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>notoriety_actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>castings_actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>period</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">5580</td><td style="border: 1px solid white;">Psycho</td><td style="border: 1px solid white;">1960.00</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">9.30</td><td style="border: 1px solid white;">10.0</td><td style="border: 1px solid white;">0.645372645372645</td><td style="border: 1px solid white;">0.012977099236641</td><td style="border: 1px solid white;">Alfred Hitchcock</td><td style="border: 1px solid white;">0.75337954939341400</td><td style="border: 1px solid white;">0.708333333333333</td><td style="border: 1px solid white;">0.245501343902328</td><td style="border: 1px solid white;">0.334586466165414</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">English</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">17065</td><td style="border: 1px solid white;">A Clockwork Orange</td><td style="border: 1px solid white;">1971.00</td><td style="border: 1px solid white;">Great Britain</td><td style="border: 1px solid white;">9.20</td><td style="border: 1px solid white;">10.0</td><td style="border: 1px solid white;">0.889434889434889</td><td style="border: 1px solid white;">0.018511450381679</td><td style="border: 1px solid white;">Stanley Kubrick</td><td style="border: 1px solid white;">0.64991334488734800</td><td style="border: 1px solid white;">0.152777777777778</td><td style="border: 1px solid white;">0.254658102136577</td><td style="border: 1px solid white;">0.150375939849624</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">English</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">1963</td><td style="border: 1px solid white;">2001: A Space Odyssey</td><td style="border: 1px solid white;">1968.00</td><td style="border: 1px solid white;">Great Britain</td><td style="border: 1px solid white;">9.10</td><td style="border: 1px solid white;">10.0</td><td style="border: 1px solid white;">0.806715806715807</td><td style="border: 1px solid white;">0.019274809160305</td><td style="border: 1px solid white;">Stanley Kubrick</td><td style="border: 1px solid white;">0.64991334488734800</td><td style="border: 1px solid white;">0.152777777777778</td><td style="border: 1px solid white;">0.138809165869436</td><td style="border: 1px solid white;">0.0413533834586466</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">English</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">6476</td><td style="border: 1px solid white;">The Shining</td><td style="border: 1px solid white;">1980.00</td><td style="border: 1px solid white;">United States, Great Britain</td><td style="border: 1px solid white;">9.10</td><td style="border: 1px solid white;">10.0</td><td style="border: 1px solid white;">0.955773955773956</td><td style="border: 1px solid white;">0.015076335877863</td><td style="border: 1px solid white;">Stanley Kubrick</td><td style="border: 1px solid white;">0.64991334488734800</td><td style="border: 1px solid white;">0.152777777777778</td><td style="border: 1px solid white;">0.415789713452691</td><td style="border: 1px solid white;">0.169172932330827</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">English</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">7025</td><td style="border: 1px solid white;">Taxi Driver</td><td style="border: 1px solid white;">1976.00</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">9.10</td><td style="border: 1px solid white;">10.0</td><td style="border: 1px solid white;">0.709254709254709</td><td style="border: 1px solid white;">0.013358778625954</td><td style="border: 1px solid white;">Martin Scorsese</td><td style="border: 1px solid white;">0.82928942807625600</td><td style="border: 1px solid white;">0.472222222222222</td><td style="border: 1px solid white;">0.832809439205505</td><td style="border: 1px solid white;">0.586466165413534</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">English</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>5</b></td><td style="border: 1px solid white;">4991</td><td style="border: 1px solid white;">The Godfather</td><td style="border: 1px solid white;">1972.00</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">9.10</td><td style="border: 1px solid white;">10.0</td><td style="border: 1px solid white;">0.684684684684685</td><td style="border: 1px solid white;">0.026335877862595</td><td style="border: 1px solid white;">Francis Ford Coppola</td><td style="border: 1px solid white;">0.43422876949740000</td><td style="border: 1px solid white;">0.347222222222222</td><td style="border: 1px solid white;">0.566671222267779</td><td style="border: 1px solid white;">0.492481203007519</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">English</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>6</b></td><td style="border: 1px solid white;">1152</td><td style="border: 1px solid white;">Once Upon a Time in America</td><td style="border: 1px solid white;">1984.00</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">9.00</td><td style="border: 1px solid white;">10.0</td><td style="border: 1px solid white;">0.665028665028665</td><td style="border: 1px solid white;">0.034351145038168</td><td style="border: 1px solid white;">Sergio Leone</td><td style="border: 1px solid white;">0.29046793760831900</td><td style="border: 1px solid white;">0.083333333333333</td><td style="border: 1px solid white;">0.667304450822288</td><td style="border: 1px solid white;">0.421052631578947</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">English</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>7</b></td><td style="border: 1px solid white;">12849</td><td style="border: 1px solid white;">Pulp Fiction</td><td style="border: 1px solid white;">1994.00</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">9.00</td><td style="border: 1px solid white;">10.0</td><td style="border: 1px solid white;">1.000000000000000</td><td style="border: 1px solid white;">0.021755725190840</td><td style="border: 1px solid white;">Quentin Tarantino</td><td style="border: 1px solid white;">0.68674176776429800</td><td style="border: 1px solid white;">0.125000000000000</td><td style="border: 1px solid white;">0.865928659286594</td><td style="border: 1px solid white;">0.541353383458647</td><td style="border: 1px solid white;">90s</td><td style="border: 1px solid white;">English</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>8</b></td><td style="border: 1px solid white;">9277</td><td style="border: 1px solid white;">Full Metal Jacket</td><td style="border: 1px solid white;">1987.00</td><td style="border: 1px solid white;">United States, Great Britain</td><td style="border: 1px solid white;">8.90</td><td style="border: 1px solid white;">10.0</td><td style="border: 1px solid white;">0.705159705159705</td><td style="border: 1px solid white;">0.014503816793893</td><td style="border: 1px solid white;">Stanley Kubrick</td><td style="border: 1px solid white;">0.64991334488734800</td><td style="border: 1px solid white;">0.152777777777778</td><td style="border: 1px solid white;">0.255751446403354</td><td style="border: 1px solid white;">0.300751879699248</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">English</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>9</b></td><td style="border: 1px solid white;">39429</td><td style="border: 1px solid white;">Gran Torino</td><td style="border: 1px solid white;">2008.00</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">8.50</td><td style="border: 1px solid white;">10.0</td><td style="border: 1px solid white;">0.834561834561835</td><td style="border: 1px solid white;">0.014503816793893</td><td style="border: 1px solid white;">Clint Eastwood</td><td style="border: 1px solid white;">0.89306759098786800</td><td style="border: 1px solid white;">0.513888888888889</td><td style="border: 1px solid white;">0.542936540476517</td><td style="border: 1px solid white;">0.195488721804511</td><td style="border: 1px solid white;">Recent</td><td style="border: 1px solid white;">English</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[26]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: search, Number of rows: 50898, Number of columns: 15</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Wow ! The results are more consistent. Top movies are Gran Torino, Pulp Fiction and The Godfather.</p>
<h1 id="Machine-Learning-:-Creating-Movies-Clusters">Machine Learning : Creating Movies Clusters<a class="anchor-link" href="#Machine-Learning-:-Creating-Movies-Clusters">&#182;</a></h1><p>As KMeans is sensible to unnormalized data, let's normalize our new predictors.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[27]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">filmtv_movies_complete</span><span class="p">[</span><span class="s2">&quot;unbiased_vote&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">normalize</span><span class="p">(</span><span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;minmax&quot;</span><span class="p">)</span>
<span class="n">drop_view</span><span class="p">(</span><span class="s2">&quot;filmtv_movies_complete_clustering&quot;</span><span class="p">)</span>
<span class="n">filmtv_movies_complete</span><span class="o">.</span><span class="n">to_db</span><span class="p">(</span><span class="s2">&quot;filmtv_movies_complete_clustering&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>The view filmtv_movies_complete_clustering was successfully dropped.
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_French</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Animation</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>avg_vote</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Romantic</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>period_90s</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>period</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Adventure</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Russian_Est_Europe</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Arabic_Middle_Est</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_German_North_Europe</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Chinese_Japan_Asian</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Italian</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>title</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>notoriety_actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>period_Old</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Fantasy</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>notoriety_director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>filmtv_id</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>votes</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Grec_Balkan</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Horror</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>country</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Others</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Drama</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Indian</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>castings_actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>castings_director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>duration</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>year</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Others</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_English</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Action</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Comedy</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Hebrew</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>unbiased_vote</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">4.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Recent</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">A.B. Stone</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">English</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Lake Placid vs. Anaconda</td><td style="border: 1px solid white;">0.0072434057673909</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00034662045060700</td><td style="border: 1px solid white;">80684</td><td style="border: 1px solid white;">0.003276003276003</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.075187969924812</td><td style="border: 1px solid white;">0E-15</td><td style="border: 1px solid white;">0.008778625954198</td><td style="border: 1px solid white;">2015.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Action</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.182208419302342</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">8.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Recent</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">A.R. Murugadoss</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Indian</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Thuppakki</td><td style="border: 1px solid white;">4.55560111157619e-05</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00008665511265200</td><td style="border: 1px solid white;">56292</td><td style="border: 1px solid white;">0E-15</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">India</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0.0150375939849624</td><td style="border: 1px solid white;">0.013888888888889</td><td style="border: 1px solid white;">0.020992366412214</td><td style="border: 1px solid white;">2012.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Action</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.241751746921569</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">8.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Recent</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">A.R. Murugadoss</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Indian</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Ghajini</td><td style="border: 1px solid white;">0.00104778825566028</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00008665511265200</td><td style="border: 1px solid white;">63738</td><td style="border: 1px solid white;">0E-15</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">India</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0.0300751879699248</td><td style="border: 1px solid white;">0.013888888888889</td><td style="border: 1px solid white;">0.027290076335878</td><td style="border: 1px solid white;">2008.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Action</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.24562354801418</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">3.90</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Recent</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Aaron Harvey</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">English</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Catch .44</td><td style="border: 1px solid white;">0.396610632772995</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00329289428076300</td><td style="border: 1px solid white;">49021</td><td style="border: 1px solid white;">0.031122031122031</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.353383458646617</td><td style="border: 1px solid white;">0E-15</td><td style="border: 1px solid white;">0.012977099236641</td><td style="border: 1px solid white;">2011.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Action</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.138777213683099</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">4.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Aaron Lipstadt</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">English</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">City Limits</td><td style="border: 1px solid white;">0.0880597694865839</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00155979202773000</td><td style="border: 1px solid white;">17196</td><td style="border: 1px solid white;">0E-15</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.203007518796992</td><td style="border: 1px solid white;">0.041666666666667</td><td style="border: 1px solid white;">0.009541984732824</td><td style="border: 1px solid white;">1985.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Action</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.379619893915945</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[27]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: filmtv_movies_complete, Number of rows: 50898, Number of columns: 38</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Let's compute the Elbow curve to find a suitable number of clusters.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[39]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="o">%</span><span class="k">matplotlib</span> inline
<span class="n">predictors</span> <span class="o">=</span> <span class="n">filmtv_movies_complete</span><span class="o">.</span><span class="n">get_columns</span><span class="p">(</span><span class="n">exclude_columns</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;avg_vote&quot;</span><span class="p">,</span>
                                                                   <span class="s2">&quot;period&quot;</span><span class="p">,</span>
                                                                   <span class="s2">&quot;director&quot;</span><span class="p">,</span>
                                                                   <span class="s2">&quot;language_area&quot;</span><span class="p">,</span>
                                                                   <span class="s2">&quot;title&quot;</span><span class="p">,</span> 
                                                                   <span class="s2">&quot;year&quot;</span><span class="p">,</span>
                                                                   <span class="s2">&quot;country&quot;</span><span class="p">,</span>
                                                                   <span class="s2">&quot;Category&quot;</span><span class="p">,</span>
                                                                   <span class="s2">&quot;filmtv_id&quot;</span><span class="p">])</span>
<span class="kn">from</span> <span class="nn">vertica_ml_python.learn.plot</span> <span class="k">import</span> <span class="n">elbow</span>
<span class="n">elbow</span> <span class="o">=</span> <span class="n">elbow</span><span class="p">(</span><span class="n">predictors</span><span class="p">,</span>
              <span class="s2">&quot;filmtv_movies_complete_clustering&quot;</span><span class="p">,</span>
              <span class="n">n_cluster</span> <span class="o">=</span> <span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">60</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>By looking at the Elbow curve, we can choose 15 clusters. Let's create a KMeans model.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[40]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.cluster</span> <span class="k">import</span> <span class="n">KMeans</span>
<span class="n">model_kmeans</span> <span class="o">=</span> <span class="n">KMeans</span><span class="p">(</span><span class="s2">&quot;filmtv_movies_clustering&quot;</span><span class="p">,</span> <span class="n">n_cluster</span> <span class="o">=</span> <span class="mi">15</span><span class="p">)</span>
<span class="n">model_kmeans</span><span class="o">.</span><span class="n">drop</span><span class="p">()</span>
<span class="n">model_kmeans</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="s2">&quot;filmtv_movies_complete_clustering&quot;</span><span class="p">,</span> <span class="n">predictors</span><span class="p">)</span>
<span class="n">model_kmeans</span><span class="o">.</span><span class="n">cluster_centers</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_french</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>category_animation</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>category_romantic</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>period_90s</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>category_adventure</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_russian_est_europe</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_german_north_europe</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_chinese_japan_asian</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_italian</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>notoriety_actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>category_others</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>category_drama</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_indian</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>castings_actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>castings_director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>duration</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>unbiased_vote</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>period_old</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>category_fantasy</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>votes</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_grec_balkan</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>category_horror</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>notoriety_director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_others</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>category_comedy</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_hebrew</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_arabic_middle_est</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_english</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>category_action</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0266596968112912</td><td style="border: 1px solid white;">0.0209095661265029</td><td style="border: 1px solid white;">0.115002613695766</td><td style="border: 1px solid white;">0.0271824359644537</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0173455524414106</td><td style="border: 1px solid white;">0.0543648719289075</td><td style="border: 1px solid white;">0.5446941975954</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.042134544411211</td><td style="border: 1px solid white;">0.113986176453505</td><td style="border: 1px solid white;">0.0128718530903461</td><td style="border: 1px solid white;">0.482109791301846</td><td style="border: 1px solid white;">0.240982749607946</td><td style="border: 1px solid white;">0.0470465237846315</td><td style="border: 1px solid white;">0.013414402855072</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0872974385781495</td><td style="border: 1px solid white;">0.0232900223862816</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.116570831155254</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">0.61011673151751</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.115434500648508</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0490272373540856</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0373528920474391</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.0217898832684825</td><td style="border: 1px solid white;">0.0953629209209795</td><td style="border: 1px solid white;">0.089497766248739</td><td style="border: 1px solid white;">0.0120385936772903</td><td style="border: 1px solid white;">0.437502195152328</td><td style="border: 1px solid white;">0.221789883268482</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0141031728580367</td><td style="border: 1px solid white;">0.0197146562905318</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0173296738126226</td><td style="border: 1px solid white;">0.00233463035019455</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0207522697795071</td><td style="border: 1px solid white;">0.0754863813229572</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.00215227333871402</td><td style="border: 1px solid white;">0.0460048426150121</td><td style="border: 1px solid white;">0.142857142857143</td><td style="border: 1px solid white;">0.0895883777239709</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.060672554532917</td><td style="border: 1px solid white;">0.284907183212268</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.138048915670937</td><td style="border: 1px solid white;">0.157564344005022</td><td style="border: 1px solid white;">0.0104383716655678</td><td style="border: 1px solid white;">0.225766287913193</td><td style="border: 1px solid white;">0.347054075867635</td><td style="border: 1px solid white;">0.0317460317460318</td><td style="border: 1px solid white;">0.0190483241330699</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0656443368307775</td><td style="border: 1px solid white;">0.0361528067444115</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.339252085014797</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.000811798132864294</td><td style="border: 1px solid white;">0.0355838181572182</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.113110539845758</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0714811104875283</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.374915437694493</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.186975014318369</td><td style="border: 1px solid white;">0.206021963649484</td><td style="border: 1px solid white;">0.011059200327198</td><td style="border: 1px solid white;">0.455839604028799</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.0595318630767149</td><td style="border: 1px solid white;">0.0191112749629865</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0741442294682722</td><td style="border: 1px solid white;">0.0438630804667347</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.206602624813963</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.929267042542286</td><td style="border: 1px solid white;">0.113796067056861</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.00102511532547412</td><td style="border: 1px solid white;">0.297478062146653</td><td style="border: 1px solid white;">0.312033714904038</td><td style="border: 1px solid white;">0.0107030647818108</td><td style="border: 1px solid white;">0.32525453086553</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0221944604312621</td><td style="border: 1px solid white;">0.00410046130189646</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0638417218384209</td><td style="border: 1px solid white;">0.000512557662737058</td><td style="border: 1px solid white;">0.999487442337263</td><td style="border: 1px solid white;">0.00205023065094823</td><td style="border: 1px solid white;">0.00410046130189646</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>5</b></td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0924753867791842</td><td style="border: 1px solid white;">0.149789029535865</td><td style="border: 1px solid white;">0.040436005625879</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0126978490835073</td><td style="border: 1px solid white;">0.0682137834036568</td><td style="border: 1px solid white;">0.417018284106892</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0651827353193109</td><td style="border: 1px solid white;">0.112830129707767</td><td style="border: 1px solid white;">0.0110997036750732</td><td style="border: 1px solid white;">0.407373861524172</td><td style="border: 1px solid white;">0.233122362869198</td><td style="border: 1px solid white;">0.0158227848101266</td><td style="border: 1px solid white;">0.00769613111385266</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0193389592123769</td><td style="border: 1px solid white;">0.0173729180225572</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.194796061884669</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0256680731364276</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>6</b></td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.000934579439252336</td><td style="border: 1px solid white;">0.022196261682243</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.0623831775700935</td><td style="border: 1px solid white;">0.00163551401869159</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0853550995271118</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.372897196261682</td><td style="border: 1px solid white;">0.000233644859813084</td><td style="border: 1px solid white;">0.164674390415291</td><td style="border: 1px solid white;">0.127660955347871</td><td style="border: 1px solid white;">0.0113208960547905</td><td style="border: 1px solid white;">0.32645508495547</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0654205607476635</td><td style="border: 1px solid white;">0.0189349927667685</td><td style="border: 1px solid white;">0.000233644859813084</td><td style="border: 1px solid white;">0.0464953271028037</td><td style="border: 1px solid white;">0.0293355496525697</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.000467289719626168</td><td style="border: 1px solid white;">0.00116822429906542</td><td style="border: 1px solid white;">0.989018691588785</td><td style="border: 1px solid white;">0.139719626168224</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>7</b></td><td style="border: 1px solid white;">0.755691282620766</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.138811771238201</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0233203775680178</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0480710519980764</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.022765130483065</td><td style="border: 1px solid white;">0.131512150726622</td><td style="border: 1px solid white;">0.0992272811401074</td><td style="border: 1px solid white;">0.0111918315100601</td><td style="border: 1px solid white;">0.32767961538466</td><td style="border: 1px solid white;">0.293725707940033</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0140235131906426</td><td style="border: 1px solid white;">0.0116601887840089</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0214932586075327</td><td style="border: 1px solid white;">0.00111049416990561</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.00666296501943365</td><td style="border: 1px solid white;">0.0210993892282066</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>8</b></td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.134818731117825</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.061681497143941</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.159947641005838</td><td style="border: 1px solid white;">0.152064451158107</td><td style="border: 1px solid white;">0.0116523921265652</td><td style="border: 1px solid white;">0.330580978993342</td><td style="border: 1px solid white;">0.46261329305136</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0180749274404864</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0291490716645636</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>9</b></td><td style="border: 1px solid white;">0.0525015441630636</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0747374922791847</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0228536133415689</td><td style="border: 1px solid white;">0.0197652872143298</td><td style="border: 1px solid white;">0.405188387893762</td><td style="border: 1px solid white;">0.339098208770846</td><td style="border: 1px solid white;">0.03613314837424</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.00988264360716492</td><td style="border: 1px solid white;">0.126978966873639</td><td style="border: 1px solid white;">0.118737560908654</td><td style="border: 1px solid white;">0.0116016153595895</td><td style="border: 1px solid white;">0.372759908152131</td><td style="border: 1px solid white;">0.638048177887585</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.010614635074178</td><td style="border: 1px solid white;">0.00247066090179123</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0185013215038489</td><td style="border: 1px solid white;">0.00123533045089561</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.00432365657813465</td><td style="border: 1px solid white;">0.0080296479308215</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">1.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>10</b></td><td style="border: 1px solid white;">0.843993085566119</td><td style="border: 1px solid white;">0.00691443388072602</td><td style="border: 1px solid white;">0.0825410544511668</td><td style="border: 1px solid white;">0.122731201382887</td><td style="border: 1px solid white;">0.0993949870354365</td><td style="border: 1px solid white;">0.0090751944684529</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0364429976768798</td><td style="border: 1px solid white;">0.167675021607606</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.00518582541054451</td><td style="border: 1px solid white;">0.126220098647656</td><td style="border: 1px solid white;">0.0975343320848939</td><td style="border: 1px solid white;">0.0108383586136823</td><td style="border: 1px solid white;">0.345137450636317</td><td style="border: 1px solid white;">0.252808988764045</td><td style="border: 1px solid white;">0.0596369922212619</td><td style="border: 1px solid white;">0.0109067974236514</td><td style="border: 1px solid white;">0.00345721694036301</td><td style="border: 1px solid white;">0.116681071737252</td><td style="border: 1px solid white;">0.0147130569856603</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.00345721694036301</td><td style="border: 1px solid white;">0.0112359550561798</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0587726879861711</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>11</b></td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.199672667757774</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0951903812546038</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.159763360939173</td><td style="border: 1px solid white;">0.127523186033824</td><td style="border: 1px solid white;">0.0107268150073087</td><td style="border: 1px solid white;">0.294847470158677</td><td style="border: 1px solid white;">0.387888707037643</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0223412968093819</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0364025363994021</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>12</b></td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0916134913400182</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0223336371923428</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0523995870972587</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0100273473108478</td><td style="border: 1px solid white;">0.0957824141027135</td><td style="border: 1px solid white;">0.126266079205915</td><td style="border: 1px solid white;">0.0125303569067616</td><td style="border: 1px solid white;">0.416371281203066</td><td style="border: 1px solid white;">0.325888787602552</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0140234292467656</td><td style="border: 1px solid white;">0.00319051959890611</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0320375089459358</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.00774840474020055</td><td style="border: 1px solid white;">0.0145852324521422</td><td style="border: 1px solid white;">0.859161349134002</td><td style="border: 1px solid white;">0.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>13</b></td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.00209205020920502</td><td style="border: 1px solid white;">0.0271966527196653</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.118200836820084</td><td style="border: 1px solid white;">0.372384937238494</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0183621699071478</td><td style="border: 1px solid white;">0.0292887029288703</td><td style="border: 1px solid white;">0.585774058577406</td><td style="border: 1px solid white;">0.0397489539748954</td><td style="border: 1px solid white;">0.075404253311102</td><td style="border: 1px solid white;">0.114103905160391</td><td style="border: 1px solid white;">0.0113254176115493</td><td style="border: 1px solid white;">0.565660955587883</td><td style="border: 1px solid white;">0.982217573221757</td><td style="border: 1px solid white;">0.0292887029288703</td><td style="border: 1px solid white;">0.00754320314989773</td><td style="border: 1px solid white;">0.0428870292887029</td><td style="border: 1px solid white;">0.0711297071129707</td><td style="border: 1px solid white;">0.0176140112977963</td><td style="border: 1px solid white;">0.0115062761506276</td><td style="border: 1px solid white;">0.0679916317991632</td><td style="border: 1px solid white;">0.0125523012552301</td><td style="border: 1px solid white;">0.0397489539748954</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>14</b></td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.011088295687885</td><td style="border: 1px solid white;">0.0331279945242984</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.031895961670089</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.1045459910071</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.366598220396988</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.135976234220047</td><td style="border: 1px solid white;">0.0826184500722489</td><td style="border: 1px solid white;">0.0114109075139373</td><td style="border: 1px solid white;">0.24816401412792</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.074880219028063</td><td style="border: 1px solid white;">0.0362725989009357</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.139630390143737</td><td style="border: 1px solid white;">0.032391526897486</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.150171115674196</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[40]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Let's add the clusters in the vDataFrame.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[43]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">model_kmeans</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">filmtv_movies_complete</span><span class="p">,</span> 
                     <span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;movies_cluster&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_French</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Animation</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>avg_vote</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Romantic</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>period_90s</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>period</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Adventure</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Russian_Est_Europe</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_German_North_Europe</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Chinese_Japan_Asian</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Italian</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>title</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>notoriety_actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Others</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Drama</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Indian</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>castings_actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>castings_director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>duration</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>year</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>unbiased_vote</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>period_Old</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Fantasy</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>filmtv_id</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>votes</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Grec_Balkan</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Horror</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>notoriety_director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>country</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Others</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Comedy</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Hebrew</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_Arabic_Middle_Est</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area_English</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category_Action</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>movies_cluster</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">7.20</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Barry Levinson</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">English</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Diner</td><td style="border: 1px solid white;">Comedy</td><td style="border: 1px solid white;">0.162088287549542</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.300751879699248</td><td style="border: 1px solid white;">0.347222222222222</td><td style="border: 1px solid white;">0.010496183206107</td><td style="border: 1px solid white;">1982.00</td><td style="border: 1px solid white;">0.394313073348633</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">18</td><td style="border: 1px solid white;">0.011466011466011</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.18977469670710600</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">11</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">7.80</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Elio Petri</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Italian</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">A ciascuno il suo</td><td style="border: 1px solid white;">Drama</td><td style="border: 1px solid white;">0.248690264680425</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.537593984962406</td><td style="border: 1px solid white;">0.152777777777778</td><td style="border: 1px solid white;">0.010114503816794</td><td style="border: 1px solid white;">1967.00</td><td style="border: 1px solid white;">0.464023324559439</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">22</td><td style="border: 1px solid white;">0.082719082719083</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.10788561525130000</td><td style="border: 1px solid white;">Italy</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">8</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">6.50</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Ray Morrison (Angelo Dorigo)</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Italian</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">A... come assassino</td><td style="border: 1px solid white;">Thriller</td><td style="border: 1px solid white;">0.00332558881144434</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.0676691729323308</td><td style="border: 1px solid white;">0E-15</td><td style="border: 1px solid white;">0.007633587786260</td><td style="border: 1px solid white;">1966.00</td><td style="border: 1px solid white;">0.325331186475542</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">24</td><td style="border: 1px solid white;">0.001638001638002</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00017331022530300</td><td style="border: 1px solid white;">Italy</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">5.50</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">Leon Klimovsky</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Italian</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">A Ghentar si muore facile</td><td style="border: 1px solid white;">Adventure</td><td style="border: 1px solid white;">0.0165368320349875</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.154135338345865</td><td style="border: 1px solid white;">0.138888888888889</td><td style="border: 1px solid white;">0.011641221374046</td><td style="border: 1px solid white;">1968.00</td><td style="border: 1px solid white;">0.283541575467166</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">30</td><td style="border: 1px solid white;">0.000819000819001</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00467937608318900</td><td style="border: 1px solid white;">Italy</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">4.00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">90s</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Carol Wiseman</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">English</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Does This Mean We are Married?</td><td style="border: 1px solid white;">Comedy</td><td style="border: 1px solid white;">0.00560338936722661</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.0676691729323308</td><td style="border: 1px solid white;">0.027777777777778</td><td style="border: 1px solid white;">0.009541984732824</td><td style="border: 1px solid white;">1990.00</td><td style="border: 1px solid white;">0.266666972816951</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">31</td><td style="border: 1px solid white;">0E-15</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00043327556325800</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">11</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[43]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: filmtv_movies_complete_clustering, Number of rows: 50898, Number of columns: 39</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We can look at the different clusters.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[44]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">filmtv_movies_complete</span><span class="o">.</span><span class="n">search</span><span class="p">(</span><span class="s2">&quot;movies_cluster = 0&quot;</span><span class="p">,</span>
                              <span class="n">usecols</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;avg_vote&quot;</span><span class="p">,</span>
                                         <span class="s2">&quot;period&quot;</span><span class="p">,</span>
                                         <span class="s2">&quot;director&quot;</span><span class="p">,</span>
                                         <span class="s2">&quot;language_area&quot;</span><span class="p">,</span>
                                         <span class="s2">&quot;title&quot;</span><span class="p">,</span> 
                                         <span class="s2">&quot;year&quot;</span><span class="p">,</span>
                                         <span class="s2">&quot;country&quot;</span><span class="p">,</span>
                                         <span class="s2">&quot;Category&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>avg_vote</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>period</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>title</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>year</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>country</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">8.00</td><td style="border: 1px solid white;">90s</td><td style="border: 1px solid white;">Koei Oguri</td><td style="border: 1px solid white;">Chinese_Japan_Asian</td><td style="border: 1px solid white;">Shi no Toge</td><td style="border: 1px solid white;">1990.00</td><td style="border: 1px solid white;">Japan</td><td style="border: 1px solid white;">Drama</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">6.90</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">Koreyoshi Kurahara</td><td style="border: 1px solid white;">Chinese_Japan_Asian</td><td style="border: 1px solid white;">Antarctica</td><td style="border: 1px solid white;">1983.00</td><td style="border: 1px solid white;">Japan</td><td style="border: 1px solid white;">Adventure</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">7.40</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">Seijun Suzuki</td><td style="border: 1px solid white;">Chinese_Japan_Asian</td><td style="border: 1px solid white;">Nikutai no mon</td><td style="border: 1px solid white;">1964.00</td><td style="border: 1px solid white;">Japan</td><td style="border: 1px solid white;">Drama</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">3.90</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">Kinji Fukasaku</td><td style="border: 1px solid white;">Chinese_Japan_Asian</td><td style="border: 1px solid white;">The Green Slime</td><td style="border: 1px solid white;">1968.00</td><td style="border: 1px solid white;">United States, Japan</td><td style="border: 1px solid white;">Fantasy</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">8.50</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">Kon Ichikawa</td><td style="border: 1px solid white;">Chinese_Japan_Asian</td><td style="border: 1px solid white;">Nobi</td><td style="border: 1px solid white;">1959.00</td><td style="border: 1px solid white;">Japan</td><td style="border: 1px solid white;">Drama</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[44]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: search, Number of rows: 1913, Number of columns: 8</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[45]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">filmtv_movies_complete</span><span class="o">.</span><span class="n">search</span><span class="p">(</span><span class="s2">&quot;movies_cluster = 1&quot;</span><span class="p">,</span>
                              <span class="n">usecols</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;avg_vote&quot;</span><span class="p">,</span>
                                         <span class="s2">&quot;period&quot;</span><span class="p">,</span>
                                         <span class="s2">&quot;director&quot;</span><span class="p">,</span>
                                         <span class="s2">&quot;language_area&quot;</span><span class="p">,</span>
                                         <span class="s2">&quot;title&quot;</span><span class="p">,</span> 
                                         <span class="s2">&quot;year&quot;</span><span class="p">,</span>
                                         <span class="s2">&quot;country&quot;</span><span class="p">,</span>
                                         <span class="s2">&quot;Category&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>avg_vote</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>period</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>title</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>year</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>country</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">4.00</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">Vera Belmont</td><td style="border: 1px solid white;">French</td><td style="border: 1px solid white;">Rouge baiser</td><td style="border: 1px solid white;">1985.00</td><td style="border: 1px solid white;">France</td><td style="border: 1px solid white;">Drama</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">4.50</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">Christine Allen</td><td style="border: 1px solid white;">French</td><td style="border: 1px solid white;">Arrêt sur image</td><td style="border: 1px solid white;">1987.00</td><td style="border: 1px solid white;">France</td><td style="border: 1px solid white;">Drama</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">7.00</td><td style="border: 1px solid white;">90s</td><td style="border: 1px solid white;">Marco Bechis</td><td style="border: 1px solid white;">Spanish_Portuguese</td><td style="border: 1px solid white;">Alambrado</td><td style="border: 1px solid white;">1991.00</td><td style="border: 1px solid white;">Italy, Argentina</td><td style="border: 1px solid white;">Drama</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">8.10</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">Marcel Carné</td><td style="border: 1px solid white;">French</td><td style="border: 1px solid white;">Le jour se lève</td><td style="border: 1px solid white;">1939.00</td><td style="border: 1px solid white;">France</td><td style="border: 1px solid white;">Drama</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">4.00</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">Jean-Louis Van Belle</td><td style="border: 1px solid white;">French</td><td style="border: 1px solid white;">A l'ombre d'un été</td><td style="border: 1px solid white;">1976.00</td><td style="border: 1px solid white;">France</td><td style="border: 1px solid white;">Drama</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[45]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: search, Number of rows: 3855, Number of columns: 8</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[46]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">filmtv_movies_complete</span><span class="o">.</span><span class="n">search</span><span class="p">(</span><span class="s2">&quot;movies_cluster = 2&quot;</span><span class="p">,</span>
                              <span class="n">usecols</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;avg_vote&quot;</span><span class="p">,</span>
                                         <span class="s2">&quot;period&quot;</span><span class="p">,</span>
                                         <span class="s2">&quot;director&quot;</span><span class="p">,</span>
                                         <span class="s2">&quot;language_area&quot;</span><span class="p">,</span>
                                         <span class="s2">&quot;title&quot;</span><span class="p">,</span> 
                                         <span class="s2">&quot;year&quot;</span><span class="p">,</span>
                                         <span class="s2">&quot;country&quot;</span><span class="p">,</span>
                                         <span class="s2">&quot;Category&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>avg_vote</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>period</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>title</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>year</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>country</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">6.50</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">Ray Morrison (Angelo Dorigo)</td><td style="border: 1px solid white;">Italian</td><td style="border: 1px solid white;">A... come assassino</td><td style="border: 1px solid white;">1966.00</td><td style="border: 1px solid white;">Italy</td><td style="border: 1px solid white;">Thriller</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">5.50</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">Leon Klimovsky</td><td style="border: 1px solid white;">Italian</td><td style="border: 1px solid white;">A Ghentar si muore facile</td><td style="border: 1px solid white;">1968.00</td><td style="border: 1px solid white;">Italy</td><td style="border: 1px solid white;">Adventure</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">4.00</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">Emilio Miraglia</td><td style="border: 1px solid white;">Italian</td><td style="border: 1px solid white;">A qualsiasi prezzo</td><td style="border: 1px solid white;">1968.00</td><td style="border: 1px solid white;">Italy</td><td style="border: 1px solid white;">Adventure</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">5.50</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">Mario Caiano</td><td style="border: 1px solid white;">Italian</td><td style="border: 1px solid white;">...a tutte le auto della polizia...</td><td style="border: 1px solid white;">1975.00</td><td style="border: 1px solid white;">Italy</td><td style="border: 1px solid white;">Thriller</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">4.00</td><td style="border: 1px solid white;">90s</td><td style="border: 1px solid white;">Romolo Guerrieri</td><td style="border: 1px solid white;">Italian</td><td style="border: 1px solid white;">A tutte le volanti</td><td style="border: 1px solid white;">1991.00</td><td style="border: 1px solid white;">Italy</td><td style="border: 1px solid white;">Thriller</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[46]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: search, Number of rows: 3717, Number of columns: 8</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[47]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">filmtv_movies_complete</span><span class="o">.</span><span class="n">search</span><span class="p">(</span><span class="s2">&quot;movies_cluster = 3&quot;</span><span class="p">,</span>
                              <span class="n">usecols</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;avg_vote&quot;</span><span class="p">,</span>
                                         <span class="s2">&quot;period&quot;</span><span class="p">,</span>
                                         <span class="s2">&quot;director&quot;</span><span class="p">,</span>
                                         <span class="s2">&quot;language_area&quot;</span><span class="p">,</span>
                                         <span class="s2">&quot;title&quot;</span><span class="p">,</span> 
                                         <span class="s2">&quot;year&quot;</span><span class="p">,</span>
                                         <span class="s2">&quot;country&quot;</span><span class="p">,</span>
                                         <span class="s2">&quot;Category&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>avg_vote</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>period</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>language_area</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>title</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>year</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>country</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Category</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">8.30</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">Sidney Lumet</td><td style="border: 1px solid white;">English</td><td style="border: 1px solid white;">Fail-Safe</td><td style="border: 1px solid white;">1964.00</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">Drama</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">4.00</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">Mark Rezyka</td><td style="border: 1px solid white;">English</td><td style="border: 1px solid white;">South of Reno</td><td style="border: 1px solid white;">1987.00</td><td style="border: 1px solid white;">Great Britain</td><td style="border: 1px solid white;">Drama</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">7.00</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">Sidney J. Furie</td><td style="border: 1px solid white;">English</td><td style="border: 1px solid white;">The Appaloosa</td><td style="border: 1px solid white;">1966.00</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">Action</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">6.00</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">Michael Miller</td><td style="border: 1px solid white;">English</td><td style="border: 1px solid white;">Can You Feel Me Dancing?</td><td style="border: 1px solid white;">1986.00</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">Drama</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">5.20</td><td style="border: 1px solid white;">Old</td><td style="border: 1px solid white;">Peter Yates</td><td style="border: 1px solid white;">English</td><td style="border: 1px solid white;">The Deep</td><td style="border: 1px solid white;">1977.00</td><td style="border: 1px solid white;">United States</td><td style="border: 1px solid white;">Adventure</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[47]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: search, Number of rows: 7391, Number of columns: 8</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Each cluster will represent films which are similar. These clusters can be used to do movie recommendations and help streaming platform to group movies in a more precise way.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Conclusion">Conclusion<a class="anchor-link" href="#Conclusion">&#182;</a></h2><p>We have solved this use-case in a pandas-like way but we never loaded the data in memory. This example showed an overview of the library. You can now start your own project by looking at the documentation first.</p>

</div>
</div>
</div>