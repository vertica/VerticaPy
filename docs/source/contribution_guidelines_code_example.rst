.. _contribution_guidelines.code.example:

============
Example
============



The vDataFrame a powerful Python object that lies at the heart of VerticaPy. vDataFrames consist of vColumn objects that represent columns in the dataset.

You can find all vDataFrame's methods inside the folder verticapy/core/vdataframe. Note that similar methods have been clubbed together inside one module/file. For examples, all methods pertaining to aggregates are in the '_aggregate.py' file.


You can define any new vDataFrame method inside these modules depending on the nature of the method. The same applies to vColumns. You can use any of the developed classes to inherit properties.

When defining a function, you should specify the 'type' hints for every variable: 

- For variables of multiple types, use the Union operator.
- For variables that are optional, use the Optional operator. 
- For variables that require literal input, use the Literal operator. 

There are examples of such hints throughout the code. 


.. code-block:: python

    @save_verticapy_logs
    def pie(
        self,
        columns: SQLColumns,
        max_cardinality: Union[None, int, tuple] = None,
        h: Union[None, int, tuple] = None,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:


Be sure to write a detailed description for each function that explains how it works.

.. code-block:: python

        """
        Draws the nested density pie chart of the input
        vDataColumns.

        Parameters
        ----------
        columns: SQLColumns
            List of the vDataColumns names.
        max_cardinality: int / tuple, optional
            Maximum number of distinct elements for
            vDataColumns 1  and  2  to be used as
            categorical. For these elements, no  h
            is picked or computed.
            If  of type tuple, represents the
            'max_cardinality' of each column.
        h: int / tuple, optional
            Interval  width  of the bar. If empty,  an
            optimized h will be computed.
            If  of type tuple, it must represent  each
            column's 'h'.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any  optional  parameter  to  pass to  the
            plotting functions.
        """

.. important:: For a detailed explaination of how to write doc-strings, please refer to :ref:`contribution_guidelines.code.auto_doc`

Important: the vDataFrame.get_columns() and vDataFrame.format_colnames() functions are essential for correctly formatting input column names.

.. ipython:: python

    from verticapy.datasets import load_titanic
    titanic = load_titanic()
    titanic.get_columns()



Use the \_genSQL method to get the current vDataFrame relation.

.. ipython:: python
    
    titanic._genSQL()


And the \_executeSQL\_ function to execute a SQL query.

.. ipython:: python

    from verticapy._utils._sql._sys import _executeSQL
    _executeSQL(f"SELECT * FROM {titanic._genSQL()} LIMIT 2")


The result of the query is accessible using one of the methods of the 'executeSQL' parameter.

.. ipython:: python

    _executeSQL(f"SELECT * FROM {titanic._genSQL()} LIMIT 2",method="fetchall")


The @save_verticapy_logs decorator saves information about a specified VerticaPy method to the QUERY_PROFILES table in the Vertica database. You can use this to collect usage statistics on methods and their parameters.

For example, to create a method to compute the correlations between two vDataFrame columns:

.. code-block:: python

    # Example correlation method for a vDataFrame

    # Add type hints + @save_verticapy_logs decorator
    @save_verticapy_logs
    def pearson(self, column1: str, column2: str):
        # Describe the function
        """
        ---------------------------------------------------------------------------
        Computes the Pearson Correlation Coefficient of the two input vColumns. 

        Parameters
        ----------
        column1: str
            Input vColumn.
        column2: str
            Input vColumn.

        Returns
        -------
        Float
            Pearson Correlation Coefficient

        See Also
        --------
        vDataFrame.corr : Computes the Correlation Matrix of the vDataFrame.
            """
        # Check data types
        # Format the columns
        column1, column2 = self.format_colnames([column1, column2])
        # Get the current vDataFrame relation
        table = self._genSQL()
        # Create the SQL statement - Label the query when possible
        query = f"SELECT /*+LABEL(vDataFrame.pearson)*/ CORR({column1}, {column2}) FROM {table};"
        # Execute the SQL query and get the result
        result = _executeSQL(query, 
                            title = "Computing Pearson coefficient", 
                            method="fetchfirstelem")
        # Return the result
        return result

Same can be done with vColumn methods.

.. code-block:: python

    # Example Method for a vColumn

    # Add types hints + @save_verticapy_logs decorator
    @save_verticapy_logs
    def pearson(self, column: str,):
        # Describe the function
        """
        ---------------------------------------------------------------------------
        Computes the Pearson Correlation Coefficient of the vColumn and the input 
        vColumn. 

        Parameters
        ----------
        column: str
            Input vColumn.

        Returns
        -------
        Float
            Pearson Correlation Coefficient

        See Also
        --------
        vDataFrame.corr : Computes the Correlation Matrix of the vDataFrame.
            """
        # Format the column
        column1 = self.parent.format_colnames([column])[0]
        # Get the current vColumn name
        column2 = self.alias
        # Get the current vDataFrame relation
        table = self.parent._genSQL()
        # Create the SQL statement - Label the query when possible
        query = f"SELECT /*+LABEL(vColumn.pearson)*/ CORR({column1}, {column2}) FROM {table};"
        # Execute the SQL query and get the result
        result = executeSQL(query, 
                            title = "Computing Pearson coefficient", 
                            method="fetchfirstelem")
        # Return the result
        return result


Functions will work exactly the same.
