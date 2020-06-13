<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Features-Engineering">Features Engineering<a class="anchor-link" href="#Features-Engineering">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>In the entire Data Preparation cycle, we need to think about the most suitable features we can use to solve the Business Problem. Many techniques are possible. We will see the most popular ones.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Customized-Features-Engineering">Customized Features Engineering<a class="anchor-link" href="#Customized-Features-Engineering">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>To build a customized feature, you can use the 'eval' method of the vDataFrame. Let's look at an example using the well-known Titanic dataset.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python</span> <span class="k">import</span> <span class="o">*</span>
<span class="n">vdf</span> <span class="o">=</span> <span class="n">vDataFrame</span><span class="p">(</span><span class="s2">&quot;titanic&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">vdf</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>fare</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>sex</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>body</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>pclass</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>age</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>name</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>cabin</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>parch</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>survived</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>boat</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>ticket</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>embarked</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>home.dest</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>sibsp</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">151.55</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2.0</td><td style="border: 1px solid white;">Allison, Miss. Helen Loraine</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">151.55</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">135</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">30.0</td><td style="border: 1px solid white;">Allison, Mr. Hudson Joshua Creighton</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">151.55</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">25.0</td><td style="border: 1px solid white;">Allison, Mrs. Hudson J C (Bessie Waldo Daniels)</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">39.0</td><td style="border: 1px solid white;">Andrews, Mr. Thomas Jr</td><td style="border: 1px solid white;">A36</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">112050</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Belfast, NI</td><td style="border: 1px solid white;">0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">49.5042</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">22</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">71.0</td><td style="border: 1px solid white;">Artagaveytia, Mr. Ramon</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">PC 17609</td><td style="border: 1px solid white;">C</td><td style="border: 1px solid white;">Montevideo, Uruguay</td><td style="border: 1px solid white;">0</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: titanic, Number of rows: 1234, Number of columns: 14
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
<p>The feature 'parch' corresponds the number of parents and children on-board. The feature 'sibsp' corresponds to the number of siblings and spouses on-board. We can create the feature 'family size' which is equal to parch + sibsp + 1.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[10]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vdf</span><span class="o">.</span><span class="n">eval</span><span class="p">(</span><span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;family_size&quot;</span><span class="p">,</span>
         <span class="n">expr</span> <span class="o">=</span> <span class="s2">&quot;parch + sibsp + 1&quot;</span><span class="p">)</span>
<span class="n">vdf</span><span class="o">.</span><span class="n">select</span><span class="p">([</span><span class="s2">&quot;parch&quot;</span><span class="p">,</span> <span class="s2">&quot;sibsp&quot;</span><span class="p">,</span> <span class="s2">&quot;family_size&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>parch</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>sibsp</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>family_size</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">4</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">4</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">4</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[10]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: titanic, Number of rows: 1234, Number of columns: 3</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>When using the 'eval' method you can enter any SQL expression and Vertica ML Python will evaluate it !</p>
<h1 id="Regular-Expressions">Regular Expressions<a class="anchor-link" href="#Regular-Expressions">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>To compute features using regular expressions, we will use the 'regexp' method.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[31]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">help</span><span class="p">(</span><span class="n">vDataFrame</span><span class="o">.</span><span class="n">regexp</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Help on function regexp in module vertica_ml_python.vdataframe:

regexp(self, column:str, pattern:str, method:str=&#39;substr&#39;, position:int=1, occurrence:int=1, replacement:str=&#39;&#39;, return_position:int=0, name:str=&#39;&#39;)
    ---------------------------------------------------------------------------
    Computes a new vcolumn based on regular expressions. 
    
    Parameters
    ----------
    column: str
            Input vcolumn used to compute the regular expression.
    pattern: str
            The regular expression.
    method: str, optional
            Method used to compute the regular expressions.
                    count     : Returns the number times a regular expression matches 
                            each element of the input vcolumn. 
                    ilike     : Returns True if the vcolumn element contains a match 
                            for the regular expression.
                    instr     : Returns the starting or ending position in a vcolumn 
                            element where a regular expression matches. 
                    like      : Returns True if the vcolumn element matches the regular 
                            expression.
                    not_ilike : Returns True if the vcolumn element does not match the 
                            case-insensitive regular expression.
                    not_like  : Returns True if the vcolumn element does not contain a 
                            match for the regular expression.
                    replace   : Replaces all occurrences of a substring that match a 
                            regular expression with another substring.
                    substr    : Returns the substring that matches a regular expression 
                            within a vcolumn.
    position: int, optional
            The number of characters from the start of the string where the function 
            should start searching for matches.
    occurrence: int, optional
            Controls which occurrence of a pattern match in the string to return.
    replacement: str, optional
            The string to replace matched substrings.
    return_position: int, optional
            Sets the position within the string to return.
    name: str, optional
            New feature name. If empty, a name will be generated.
    
    Returns
    -------
    vDataFrame
            self
    
    See Also
    --------
    vDataFrame.eval : Evaluates a customized expression.

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
<p>Let's consider the following example. We can notice that the passengers title is included on each passenger name.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[12]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vdf</span><span class="p">[</span><span class="s2">&quot;name&quot;</span><span class="p">]</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>name</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">Allison, Miss. Helen Loraine</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">Allison, Mr. Hudson Joshua Creighton</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">Allison, Mrs. Hudson J C (Bessie Waldo Daniels)</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">Andrews, Mr. Thomas Jr</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">Artagaveytia, Mr. Ramon</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[12]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: name, Number of rows: 1234, dtype: varchar(164)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Let's extract it using regular expressions.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[15]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vdf</span><span class="o">.</span><span class="n">regexp</span><span class="p">(</span><span class="n">column</span> <span class="o">=</span> <span class="s2">&quot;name&quot;</span><span class="p">,</span>
           <span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;title&quot;</span><span class="p">,</span>
           <span class="n">pattern</span> <span class="o">=</span> <span class="s2">&quot; ([A-Za-z])+\.&quot;</span><span class="p">,</span>
           <span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;substr&quot;</span><span class="p">)</span>
<span class="n">vdf</span><span class="o">.</span><span class="n">select</span><span class="p">([</span><span class="s2">&quot;name&quot;</span><span class="p">,</span> <span class="s2">&quot;title&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>name</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>title</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">Allison, Miss. Helen Loraine</td><td style="border: 1px solid white;"> Miss.</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">Allison, Mr. Hudson Joshua Creighton</td><td style="border: 1px solid white;"> Mr.</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">Allison, Mrs. Hudson J C (Bessie Waldo Daniels)</td><td style="border: 1px solid white;"> Mrs.</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">Andrews, Mr. Thomas Jr</td><td style="border: 1px solid white;"> Mr.</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">Artagaveytia, Mr. Ramon</td><td style="border: 1px solid white;"> Mr.</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[15]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: titanic, Number of rows: 1234, Number of columns: 2</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Advanced-Analytical-Functions">Advanced Analytical Functions<a class="anchor-link" href="#Advanced-Analytical-Functions">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Vertica ML Python advanced analytical functions are managed by the 'analytic' method.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[17]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">help</span><span class="p">(</span><span class="n">vDataFrame</span><span class="o">.</span><span class="n">analytic</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Help on function analytic in module vertica_ml_python.vdataframe:

analytic(self, func:str, column:str=&#39;&#39;, by:list=[], order_by=[], column2:str=&#39;&#39;, name:str=&#39;&#39;, offset:int=1, x_smoothing:float=0.5, add_count:bool=True)
    ---------------------------------------------------------------------------
    Adds a new vcolumn to the vDataFrame by using an advanced analytical function 
    on one or two specific vcolumns.
    
    Parameters
    ----------
    func: str
            Function to apply.
                    beta         : Beta Coefficient between 2 vcolumns
                    count        : number of non-missing elements
                    corr         : Pearson correlation between 2 vcolumns
                    cov          : covariance between 2 vcolumns
                    dense_rank   : dense rank
                    ema          : exponential moving average
                    first_value  : first non null lead
                    iqr          : interquartile range
                    kurtosis     : kurtosis
                    jb           : Jarque Bera index 
                    lead         : next element
                    lag          : previous element
                    last_value   : first non null lag
                    mad          : median absolute deviation
                    mae          : mean absolute error (deviation)
                    max          : maximum
                    mean         : average
                    median       : median
                    min          : minimum
                    mode         : most occurent element
                    q%           : q quantile (ex: 50% for the median)
                    pct_change   : ratio between the current value and the previous one
                    percent_rank : percent rank
                    prod         : product
                    range        : difference between the max and the min
                    rank         : rank
                    row_number   : row number
                    sem          : standard error of the mean
                    skewness     : skewness
                    sum          : sum
                    std          : standard deviation
                    unique       : cardinality (count distinct)
                    var          : variance
                            Other analytical functions could work if it is part of 
                            the DB version you are using.
    column: str, optional
            Input vcolumn.
    by: list, optional
            vcolumns used in the partition.
    order_by: dict / list, optional
            List of the vcolumns used to sort the data using asc order or
            dictionary of all the sorting methods. For example, to sort by &#34;column1&#34;
            ASC and &#34;column2&#34; DESC, write {&#34;column1&#34;: &#34;asc&#34;, &#34;column2&#34;: &#34;desc&#34;}
    column2: str, optional
            Second input vcolumn in case of functions using 2 parameters.
    name: str, optional
            Name of the new vcolumn. If empty a default name based on the other
            parameters will be generated.
    offset: int, optional
            Lead/Lag offset if parameter &#39;func&#39; is the function &#39;lead&#39;/&#39;lag&#39;.
    x_smoothing: float, optional
            The smoothing parameter of the &#39;ema&#39; if the function is &#39;ema&#39;. It must be in [0;1]
    add_count: bool, optional
            If the function is the &#39;mode&#39; and this parameter is True then another column will 
            be added to the vDataFrame with the mode number of occurences.
    
    Returns
    -------
    vDataFrame
            self
    
    See Also
    --------
    vDataFrame.eval    : Evaluates a customized expression.
    vDataFrame.rolling : Computes a customized moving window.

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
<p>Many different techniques are available. Let's use the 'USA 2015 Flights' datasets to do some computations.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[25]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python</span> <span class="k">import</span> <span class="o">*</span>
<span class="n">vdf</span> <span class="o">=</span> <span class="n">vDataFrame</span><span class="p">(</span><span class="s2">&quot;flights&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">vdf</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>departure_delay</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>origin_airport</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>scheduled_departure</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>airline</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>destination_airport</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>arrival_delay</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">6</td><td style="border: 1px solid white;">MDW</td><td style="border: 1px solid white;">2015-05-06 21:30:00</td><td style="border: 1px solid white;">WN</td><td style="border: 1px solid white;">CMH</td><td style="border: 1px solid white;">-8</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">ORD</td><td style="border: 1px solid white;">2015-05-06 21:40:00</td><td style="border: 1px solid white;">MQ</td><td style="border: 1px solid white;">CMH</td><td style="border: 1px solid white;">-13</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">23</td><td style="border: 1px solid white;">BWI</td><td style="border: 1px solid white;">2015-05-06 22:15:00</td><td style="border: 1px solid white;">WN</td><td style="border: 1px solid white;">CMH</td><td style="border: 1px solid white;">28</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">-1</td><td style="border: 1px solid white;">ATL</td><td style="border: 1px solid white;">2015-05-06 22:20:00</td><td style="border: 1px solid white;">WN</td><td style="border: 1px solid white;">CMH</td><td style="border: 1px solid white;">0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">-7</td><td style="border: 1px solid white;">ATL</td><td style="border: 1px solid white;">2015-05-06 22:22:00</td><td style="border: 1px solid white;">DL</td><td style="border: 1px solid white;">CMH</td><td style="border: 1px solid white;">-16</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: flights, Number of rows: 4068736, Number of columns: 6
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
<p>For each flight, let's compute the previous departure delay for the same airline.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[26]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vdf</span><span class="o">.</span><span class="n">analytic</span><span class="p">(</span><span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;previous_departure_delay&quot;</span><span class="p">,</span>
             <span class="n">func</span> <span class="o">=</span> <span class="s2">&quot;lag&quot;</span><span class="p">,</span>
             <span class="n">column</span> <span class="o">=</span> <span class="s2">&quot;departure_delay&quot;</span><span class="p">,</span>
             <span class="n">by</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;airline&quot;</span><span class="p">,</span> <span class="s2">&quot;destination_airport&quot;</span><span class="p">,</span> <span class="s2">&quot;origin_airport&quot;</span><span class="p">],</span>
             <span class="n">order_by</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;scheduled_departure&quot;</span><span class="p">:</span> <span class="s2">&quot;asc&quot;</span><span class="p">})</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>departure_delay</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>origin_airport</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>scheduled_departure</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>airline</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>destination_airport</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>arrival_delay</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>previous_departure_delay</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">-3</td><td style="border: 1px solid white;">10397</td><td style="border: 1px solid white;">2015-10-01 10:27:00</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">-14</td><td style="border: 1px solid white;">None</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">10397</td><td style="border: 1px solid white;">2015-10-01 14:44:00</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">-1</td><td style="border: 1px solid white;">-3</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">12</td><td style="border: 1px solid white;">10397</td><td style="border: 1px solid white;">2015-10-02 10:27:00</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">4</td><td style="border: 1px solid white;">0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">-4</td><td style="border: 1px solid white;">10397</td><td style="border: 1px solid white;">2015-10-02 14:44:00</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">-6</td><td style="border: 1px solid white;">12</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">-4</td><td style="border: 1px solid white;">10397</td><td style="border: 1px solid white;">2015-10-03 10:27:00</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">-2</td><td style="border: 1px solid white;">-4</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[26]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: flights, Number of rows: 4068736, Number of columns: 7</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Moving-Windows">Moving Windows<a class="anchor-link" href="#Moving-Windows">&#182;</a></h1><p>Moving windows are powerful features. They can bring a lot of information. Moving windows are managed by the 'rolling' method in Vertica ML Python.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[27]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">help</span><span class="p">(</span><span class="n">vDataFrame</span><span class="o">.</span><span class="n">rolling</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Help on function rolling in module vertica_ml_python.vdataframe:

rolling(self, func:str, column:str, preceding, following, column2:str=&#39;&#39;, name:str=&#39;&#39;, by:list=[], order_by=[], method:str=&#39;rows&#39;, rule:str=&#39;auto&#39;)
    ---------------------------------------------------------------------------
    Adds a new vcolumn to the vDataFrame by using an advanced analytical window 
    function on one or two specific vcolumns.
    
    Parameters
    ----------
    func: str
            Function to use.
                    beta        : Beta Coefficient between 2 vcolumns
                    count       : number of non-missing elements
                    corr        : Pearson correlation between 2 vcolumns
                    cov         : covariance between 2 vcolumns
                    kurtosis    : kurtosis
                    jb          : Jarque Bera index 
                    mae         : mean absolute error (deviation)
                    max         : maximum
                    mean        : average
                    min         : minimum
                    prod        : product
                    range       : difference between the max and the min
                    sem         : standard error of the mean
                    skewness    : skewness
                    sum         : sum
                    std         : standard deviation
                    var         : variance
                            Other window functions could work if it is part of 
                            the DB version you are using.
    column: str
            Input vcolumn.
    preceding: int/str
            First part of the moving window. With which lag/lead the window 
            should begin. It can be an integer or an interval.
    following: int/str
            Second part of the moving window. With which lag/lead the window 
            should end. It can be an integer or an interval.
    column2: str, optional
            Second input vcolumn in case of functions using 2 parameters.
    name: str, optional
            Name of the new vcolumn. If empty a default name based on the other
            parameters will be generated.
    by: list, optional
            vcolumns used in the partition.
    order_by: dict / list, optional
            List of the vcolumns used to sort the data using asc order or
            dictionary of all the sorting methods. For example, to sort by &#34;column1&#34;
            ASC and &#34;column2&#34; DESC, write {&#34;column1&#34;: &#34;asc&#34;, &#34;column2&#34;: &#34;desc&#34;}
    method: str, optional
            Method used to compute the window.
                    rows : Uses number of leads/lags instead of time intervals
                    range: Uses time intervals instead of number of leads/lags
    rule: str, optional
            Rule used to compute the window.
                    auto   : The &#39;preceding&#39; parameter will correspond to a past event and 
                            the parameter &#39;following&#39; to a future event.
                    past   : Both parameters &#39;preceding&#39; and following will consider
                            past events.
                    future : Both parameters &#39;preceding&#39; and following will consider
                            future events.
    
    Returns
    -------
    vDataFrame
            self
    
    See Also
    --------
    vDataFrame.eval     : Evaluates a customized expression.
    vDataFrame.analytic : Adds a new vcolumn to the vDataFrame by using an advanced 
            analytical function on a specific vcolumn.

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
<p>Let's for example compute the number of flights that the same airline has to manage two hours preceding the concerned flight and one hour following.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[30]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vdf</span><span class="o">.</span><span class="n">rolling</span><span class="p">(</span><span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;number_flights_to_manage_by_airline_2hp_1hf&quot;</span><span class="p">,</span>
            <span class="n">func</span> <span class="o">=</span> <span class="s2">&quot;count&quot;</span><span class="p">,</span>
            <span class="n">column</span> <span class="o">=</span> <span class="s2">&quot;airline&quot;</span><span class="p">,</span>
            <span class="n">by</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;airline&quot;</span><span class="p">,</span> <span class="s2">&quot;origin_airport&quot;</span><span class="p">],</span>
            <span class="n">order_by</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;scheduled_departure&quot;</span><span class="p">:</span> <span class="s2">&quot;asc&quot;</span><span class="p">},</span>
            <span class="n">preceding</span> <span class="o">=</span> <span class="s2">&quot;2 hours&quot;</span><span class="p">,</span>
            <span class="n">following</span> <span class="o">=</span> <span class="s2">&quot;1 hour&quot;</span><span class="p">,</span>
            <span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;range&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>departure_delay</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>origin_airport</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>scheduled_departure</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>airline</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>destination_airport</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>arrival_delay</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>previous_departure_delay</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>number_flights_to_manage_by_airline_2hp_1hf</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">-11</td><td style="border: 1px solid white;">10140</td><td style="border: 1px solid white;">2015-10-01 10:55:00</td><td style="border: 1px solid white;">AA</td><td style="border: 1px solid white;">11298</td><td style="border: 1px solid white;">-20</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">-10</td><td style="border: 1px solid white;">10140</td><td style="border: 1px solid white;">2015-10-01 12:09:00</td><td style="border: 1px solid white;">AA</td><td style="border: 1px solid white;">11298</td><td style="border: 1px solid white;">-18</td><td style="border: 1px solid white;">-11</td><td style="border: 1px solid white;">2</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">-2</td><td style="border: 1px solid white;">10140</td><td style="border: 1px solid white;">2015-10-01 14:20:00</td><td style="border: 1px solid white;">AA</td><td style="border: 1px solid white;">11298</td><td style="border: 1px solid white;">-11</td><td style="border: 1px solid white;">-10</td><td style="border: 1px solid white;">1</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">-6</td><td style="border: 1px solid white;">10140</td><td style="border: 1px solid white;">2015-10-01 16:19:00</td><td style="border: 1px solid white;">AA</td><td style="border: 1px solid white;">11298</td><td style="border: 1px solid white;">5</td><td style="border: 1px solid white;">-2</td><td style="border: 1px solid white;">2</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">-9</td><td style="border: 1px solid white;">10140</td><td style="border: 1px solid white;">2015-10-02 10:55:00</td><td style="border: 1px solid white;">AA</td><td style="border: 1px solid white;">11298</td><td style="border: 1px solid white;">-28</td><td style="border: 1px solid white;">-6</td><td style="border: 1px solid white;">1</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[30]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: flights, Number of rows: 4068736, Number of columns: 8</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Moving windows give us infinite number of possibilities to create new features. When the Data Preparation is finished, it is time to create a Machine Learning Model. Our next lesson will introduce the different types of ML algorithms !</p>

</div>
</div>
</div>