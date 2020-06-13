<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="unsupervised.fit">unsupervised.fit<a class="anchor-link" href="#unsupervised.fit">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">unsupervised</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">input_relation</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
                 <span class="n">X</span><span class="p">:</span> <span class="nb">list</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Trains the model.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">input_relation</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Train relation.</td> </tr>
    <tr> <td><div class="param_name">X</div></td> <td><div class="type">list</div></td> <td><div class = "no">&#10060;</div></td> <td>List of the predictors.</td> </tr>
</table><h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>object</b> : self</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Example">Example<a class="anchor-link" href="#Example">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[9]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.cluster</span> <span class="k">import</span> <span class="n">KMeans</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">KMeans</span><span class="p">(</span><span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;public.KMeans_iris&quot;</span><span class="p">,</span>
               <span class="n">n_cluster</span> <span class="o">=</span> <span class="mi">3</span><span class="p">)</span>
<span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="s2">&quot;public.iris&quot;</span><span class="p">,</span> <span class="p">[</span><span class="s2">&quot;PetalLengthCm&quot;</span><span class="p">,</span> <span class="s2">&quot;SepalLengthCm&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[9]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>

=======
centers
=======
petallengthcm|sepallengthcm
-------------+-------------
   4.39310   |   5.87414   
   1.49412   |   5.00784   
   5.67805   |   6.83902   


=======
metrics
=======
Evaluation metrics:
     Total Sum of Squares: 566.03207
     Within-Cluster Sum of Squares: 
         Cluster 0: 23.508448
         Cluster 1: 9.885098
         Cluster 2: 20.407805
     Total Within-Cluster Sum of Squares: 53.801351
     Between-Cluster Sum of Squares: 512.23072
     Between-Cluster SS / Total SS: 90.49%
 Number of iterations performed: 5
 Converged: True
 Call:
kmeans(&#39;public.KMeans_iris&#39;, &#39;public.iris&#39;, &#39;&#34;PetalLengthCm&#34;, &#34;SepalLengthCm&#34;&#39;, 3
USING PARAMETERS max_iterations=300, epsilon=0.0001, init_method=&#39;kmeanspp&#39;, distance_method=&#39;euclidean&#39;)</pre>
</div>

</div>

</div>
</div>

</div>