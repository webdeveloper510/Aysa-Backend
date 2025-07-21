"""
# API to inference product trained model
class ProductSemanticSearchView(APIView):

    def post(self , request , format=None):
        try:
            
            user_query = request.data.get("query")
            if not user_query:
                return BAD_RESPONSE("user query is required ")
            
            VectoDB_Path = os.path.join(os.getcwd() , "VectorDBS", "Product_DB", "faiss_index")

            # Call Product Inference Model
            # Function to inference model
            inference_obj = UserInference(VectoDB_Path)

            # STEP 1 
            retriever = inference_obj.LoadVectorDB()

            result_dict = inference_obj.ModelInference(retriever, user_query)

            return Success_RESPONSE(user_query , result_dict)


        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"[ERROR] Occur Reason: {str(e)} (line {exc_tb.tb_lineno})"
            print(error_message)
            return []

"""