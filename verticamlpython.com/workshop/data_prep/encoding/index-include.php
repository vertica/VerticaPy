<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Encoding">Encoding<a class="anchor-link" href="#Encoding">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Encoding features can be very important in the Data Science cycle. Indeed, Data Science loves generality and having too many categories can lead to unexpected results. Besides, some algorithms optimizations are linear and they prefer categorized information. Some others can not proceed non-numerical features.</p>
<p>There are many encoding techniques:</p>
<ul>
<li><b>User-Defined Encoding</b> : The most flexible encoding. The user choose to encode the different categories the way he/she wants.</li>
<li><b>Label Encoding</b> : Each category is converted to an integer using a bijection to [0;n-1] where n is the feature number of unique values.</li>
<li><b>One Hot Encoding</b> : It creates dummies (values in {0,1}) of each categories. All the categories are then separated into n features.</li>
<li><b>Mean Encoding</b> : It uses the frequencies of each category regarding a specific response column.</li>
<li><b>Discretization</b> : It uses different mathematical techniques to encode continuous features into categories.</li>
</ul>
<p>To see how to encode data in Vertica ML Python, we will use the well-known 'Titanic' dataset.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[71]:</div>
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
<p>Let's look at the 'age' of the Titanic passengers.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[73]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vdf</span><span class="p">[</span><span class="s2">&quot;age&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">hist</span><span class="p">()</span>
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
<p>By using the 'discretize' method, it is possible to discretize it using same width bins.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[41]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vdf</span><span class="p">[</span><span class="s2">&quot;age&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">discretize</span><span class="p">(</span><span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;same_width&quot;</span><span class="p">,</span> <span class="n">h</span> <span class="o">=</span> <span class="mi">10</span><span class="p">)</span>
<span class="n">vdf</span><span class="p">[</span><span class="s2">&quot;age&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">hist</span><span class="p">()</span>
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
<p>It is also possible to discretize it using the same frequence per bin.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[44]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vdf</span><span class="p">[</span><span class="s2">&quot;age&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">discretize</span><span class="p">(</span><span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;same_freq&quot;</span><span class="p">,</span> <span class="n">bins</span> <span class="o">=</span> <span class="mi">5</span><span class="p">)</span>
<span class="n">vdf</span><span class="p">[</span><span class="s2">&quot;age&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">hist</span><span class="p">()</span>
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
<p>Computing categories using a response column can also be a good solution.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[46]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vdf</span><span class="p">[</span><span class="s2">&quot;age&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">discretize</span><span class="p">(</span><span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;smart&quot;</span><span class="p">,</span> <span class="n">response</span> <span class="o">=</span> <span class="s2">&quot;survived&quot;</span><span class="p">,</span> <span class="n">bins</span> <span class="o">=</span> <span class="mi">6</span><span class="p">)</span>
<span class="n">vdf</span><span class="p">[</span><span class="s2">&quot;age&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;avg&quot;</span><span class="p">,</span> <span class="n">of</span> <span class="o">=</span> <span class="s2">&quot;survived&quot;</span><span class="p">)</span>
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
<p>Many techniques are available when using the 'discretize' method.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[47]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">help</span><span class="p">(</span><span class="n">vdf</span><span class="p">[</span><span class="s2">&quot;embarked&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">discretize</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Help on method discretize in module vertica_ml_python.vcolumn:

discretize(method:str=&#39;auto&#39;, h:float=0, bins:int=-1, k:int=6, new_category:str=&#39;Others&#39;, response:str=&#39;&#39;, min_bin_size:int=20, return_enum_trans:bool=False) method of vertica_ml_python.vcolumn.vColumn instance
    ---------------------------------------------------------------------------
    Discretizes the vcolumn using the input method.
    
    Parameters
    ----------
    method: str, optional
            The method used to discretize the vcolumn.
                    auto       : Uses method &#39;same_width&#39; for numerical vcolumns, cast 
                            the other types to varchar.
                    same_freq  : Computes bins with the same number of elements.
                    same_width : Computes regular width bins.
                    smart      : Uses the Random Forest on a response column to find the most 
                            relevant interval to use for the discretization.
                    topk       : Keeps the topk most frequent categories and merge the other 
                            into one unique category.
    h: float, optional
            The interval size to convert used to convert the vcolumn. If this parameter 
            is equal to 0, an optimised interval will be computed.
    bins: int, optional
            Number of bins used for the discretization (must be &gt; 1)
    k: int, optional
            The integer k of the &#39;topk&#39; method.
    new_category: str, optional
            The name of the merging category when using the &#39;topk&#39; method.
    response: str, optional
            Response vcolumn when using the &#39;smart&#39; method.
    min_bin_size: int, optional
            Minimum Number of elements in the bin when using the &#39;smart&#39; method.
    return_enum_trans: bool, optional
            Returns the transformation instead of the vDataFrame parent and do not apply
            it. This parameter is very useful for testing to be able to look at the final 
            transformation.
    
    Returns
    -------
    vDataFrame
            self.parent
    
    See Also
    --------
    vDataFrame[].decode       : Encodes the vcolumn using a user defined Encoding.
    vDataFrame[].get_dummies  : Encodes the vcolumn using the One Hot Encoding.
    vDataFrame[].label_encode : Encodes the vcolumn using the Label Encoding.
    vDataFrame[].mean_encode  : Encodes the vcolumn using the Mean Encoding of a response.

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
<p>To easily encode a categorical feature, it is possible to use the Label Encoding. For example, the column 'sex' has two categories (male and female). We can represent them respectively by 0 and 1.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[51]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vdf</span><span class="p">[</span><span class="s2">&quot;sex&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">label_encode</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span><span class="n">vdf</span><span class="p">[</span><span class="s2">&quot;sex&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>sex</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">1</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">1</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">1</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: sex, Number of rows: 1234, dtype: int
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
<p>When a feature has few categories, the most suitable choice is the One Hot Encoding. Indeed, Label Encoding convert a categorical feature to numerical without keeping a real mathematical sense. Let's use a One Hot Encoding on the 'embarked' column.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[53]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vdf</span><span class="p">[</span><span class="s2">&quot;embarked&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">get_dummies</span><span class="p">()</span>
<span class="n">vdf</span><span class="o">.</span><span class="n">select</span><span class="p">([</span><span class="s2">&quot;embarked&quot;</span><span class="p">,</span> <span class="s2">&quot;embarked_C&quot;</span><span class="p">,</span> <span class="s2">&quot;embarked_Q&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>embarked</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>embarked_C</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>embarked_Q</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">C</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[53]:</div>




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
<p>The One Hot Encoding can be expensive if the number of categories of the concerned column is big. That's why the 'Mean Encoding' can be a solution in this case. 'Mean encoding' will replace each category of a variable by its corresponding average over a partition by a response column. It is an efficient way to encode the data but be careful as it can lead to over-fitting.</p>
<p>Let's use a 'Mean Encoding' on the 'home.dest' variable.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[54]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vdf</span><span class="p">[</span><span class="s2">&quot;home.dest&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">mean_encode</span><span class="p">(</span><span class="s2">&quot;survived&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">vdf</span><span class="p">[</span><span class="s2">&quot;home.dest&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>The mean encoding was successfully done.
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>home.dest</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">1.0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">0.5</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">0.5</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">0.0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">0.0</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: home.dest, Number of rows: 1234, dtype: int
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
<p>Many other different encoding techniques are available in Vertica ML Python. For example, the 'case_when' and 'decode' methods allow the user to proceed a customized encoding on a column. The 'discretize' method allows also to reduce the number of categories of a specific column. Mastering other techniques can be relevant as many problems will occur when dealing with different data sources. In our next lesson, we will see how to normalize the data and why it is very important.</p>

</div>
</div>
</div>