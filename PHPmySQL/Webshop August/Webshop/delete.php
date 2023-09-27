<html>
   <link href="style.css" rel="stylesheet">
   <head>
      <title>Delete a Record from MySQL Database</title>
   </head>
   
   <body>
      <?php

require('db.php');
         if(isset($_POST['delete'])) {
            
            
            if(! $con ) {
               die('Could not connect: ' );
            }
				
            $emp_id = $_POST['emp_id'];
            
            $sql = "DELETE FROM `users` WHERE id = $emp_id" ;
            $retval = mysqli_query( $con, $sql );
            
            if(! $retval ) {
               die('Could not delete data: ' );
            }
            
            echo "Deleted data successfully\n";
            
            mysqli_close($con);
         }else {
            ?>
               <form method = "post" action = "<?php $_PHP_SELF ?>" class="form">
                  <table width = "400" cellspacing = "1" 
                     cellpadding = "2">
                     <h1 class="login-title">Delete Users</h1>
                     <tr>
                        <td width = "100">User ID</td>
                        <td><input name = "emp_id" type = "number" 
                           id = "emp_id" placeholder="Select Id to be deleted"></td>
                     </tr>
                     
                     <tr>
                        <td width = "100"> </td>
                        <td> </td><br/>
                     </tr>
                     
                     <tr>
                        <td width = "100"> </td>
                        <td>
                           <input name = "delete" type = "submit" 
                              id = "delete" value = "Delete">
                        </td>
                     </tr>
                     
                  </table>
               </form>
            <?php
         }
      ?>
      
   </body>
</html>
