<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="LinearSVR">LinearSVR<a class="anchor-link" href="#LinearSVR">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">LinearSVR</span><span class="p">(</span><span class="n">name</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
          <span class="n">cursor</span> <span class="o">=</span> <span class="kc">None</span><span class="p">,</span>
          <span class="n">tol</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">1e-4</span><span class="p">,</span> 
          <span class="n">C</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">1.0</span><span class="p">,</span> 
          <span class="n">fit_intercept</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">True</span><span class="p">,</span>
          <span class="n">intercept_scaling</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">1.0</span><span class="p">,</span>
          <span class="n">intercept_mode</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;regularized&quot;</span><span class="p">,</span>
          <span class="n">acceptable_error_margin</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">0.1</span><span class="p">,</span>
          <span class="n">max_iter</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">100</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Creates a LinearSVR object by using the Vertica Highly Distributed and Scalable SVM on the data. This algorithm will find the hyperplan which will approximate the data distribution.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">name</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Name of the the model. The model will be stored in the DB.</td> </tr>
    <tr> <td><div class="param_name">cursor</div></td> <td><div class="type">DBcursor</div></td> <td><div class = "yes">&#10003;</div></td> <td>Vertica DB cursor.</td> </tr>
    <tr> <td><div class="param_name">tol</div></td> <td><div class="type">float</div></td> <td><div class = "yes">&#10003;</div></td> <td>Used to control accuracy.</td> </tr>
    <tr> <td><div class="param_name">C</div></td> <td><div class="type">float</div></td> <td><div class = "yes">&#10003;</div></td> <td>The weight for misclassification cost. The algorithm minimizes the regularization cost and the misclassification cost.</td> </tr>
    <tr> <td><div class="param_name">fit_intercept</div></td> <td><div class="type">bool</div></td> <td><div class = "yes">&#10003;</div></td> <td>A bool to fit also the intercept.</td> </tr>
    <tr> <td><div class="param_name">intercept_scaling</div></td> <td><div class="type">float</div></td> <td><div class = "yes">&#10003;</div></td> <td>A float value, serves as the value of a dummy feature whose coefficient Vertica uses to calculate the model intercept. Because the dummy feature is not in the training data, its values are set to a constant, by default set to 1. </td> </tr>
    <tr> <td><div class="param_name">intercept_mode</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Specify how to treat the intercept.<br><ul><li><b>regularized :</b> Fits the intercept and applies a regularization on it.</li><li><b>unregularized :</b> Fits the intercept but does not include it in regularization.</li></ul></td></tr>
    <tr> <td><div class="param_name">acceptable_error_margin</div></td> <td><div class="type">float</div></td> <td><div class = "yes">&#10003;</div></td> <td>Defines the acceptable error margin. Any data points outside this region add a penalty to the cost function.</td> </tr>
    <tr> <td><div class="param_name">max_iter</div></td> <td><div class="type">int</div></td> <td><div class = "yes">&#10003;</div></td> <td>The maximum number of iterations that the algorithm performs.</td> </tr>
</table><h3 id="Attributes">Attributes<a class="anchor-link" href="#Attributes">&#182;</a></h3><p>After the object creation, all the parameters become attributes. The model will also create extra attributes when fitting the model:</p>
<table id="parameters">
    <tr> <th>Name</th> <th>Type</th>  <th>Description</th> </tr>
    <tr> <td><div class="param_name">coef</div></td> <td><div class="type">tablesample</div></td> <td>Coefficients and their mathematical information (pvalue, std, value...)</td> </tr>
    <tr> <td><div class="param_name">input_relation</div></td> <td><div class="type">str</div></td> <td>Train relation.</td> </tr>
    <tr> <td><div class="param_name">X</div></td> <td><div class="type">list</div></td> <td>List of the predictors.</td> </tr>
    <tr> <td><div class="param_name">y</div></td> <td><div class="type">str</div></td> <td>Response column.</td> </tr>
    <tr> <td><div class="param_name">test_relation</div></td> <td><div class="type">float</div></td> <td>Relation used to test the model. All the model methods are abstractions which will simplify the process. The test relation will be used by many methods to evaluate the model. If empty, the train relation will be used as test. You can change it anytime by changing the test_relation attribute of the object.</td> </tr>
</table><h3 id="Methods">Methods<a class="anchor-link" href="#Methods">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Description</th> </tr>
    <tr> <td><a href="../Regression/deploySQL/index.php">deploySQL</a></td> <td>Returns the SQL code needed to deploy the model.</td> </tr>
    <tr> <td><a href="../Regression/drop/index.php">drop</a></td> <td>Drops the model from the Vertica DB.</td> </tr>
    <tr> <td><a href="../Regression/features_importance/index.php">features_importance</a></td> <td>Computes the model features importance by normalizing the LinearSVC hyperplan coefficients.</td> </tr>
    <tr> <td><a href="../Regression/fit/index.php">fit</a></td> <td>Trains the model.</td> </tr>
    <tr> <td><a href="../Regression/plot/index.php">plot</a></td> <td>Draws the LinearSVR if the number of predictors is lesser than 2.</td> </tr>
    <tr> <td><a href="../Regression/predict/index.php">predict</a></td> <td>Predicts using the input relation.</td> </tr>
    <tr> <td><a href="../Regression/regression_report/index.php">regression_report</a></td> <td>Computes a regression report using multiple metrics to evaluate the model (r2, mse, max error...). </td> </tr>
    <tr> <td><a href="../Regression/score/index.php">score</a></td> <td>Computes the model score.</td> </tr>

</table>
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
<div class="prompt input_prompt">In&nbsp;[10]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.svm</span> <span class="k">import</span> <span class="n">LinearSVR</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">LinearSVR</span><span class="p">(</span><span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;winequality_svm&quot;</span><span class="p">,</span>
                  <span class="n">tol</span> <span class="o">=</span> <span class="mf">1e-4</span><span class="p">,</span> 
                  <span class="n">C</span> <span class="o">=</span> <span class="mf">1.0</span><span class="p">,</span> 
                  <span class="n">fit_intercept</span> <span class="o">=</span> <span class="kc">True</span><span class="p">,</span>
                  <span class="n">intercept_scaling</span> <span class="o">=</span> <span class="mf">1.0</span><span class="p">,</span>
                  <span class="n">intercept_mode</span> <span class="o">=</span> <span class="s2">&quot;regularized&quot;</span><span class="p">,</span>
                  <span class="n">acceptable_error_margin</span> <span class="o">=</span> <span class="mf">0.1</span><span class="p">,</span>
                  <span class="n">max_iter</span> <span class="o">=</span> <span class="mi">100</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">model</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;LinearSVR&gt;
</pre>
</div>
</div>

</div>
</div>

</div>