from rest_framework.response import Response
from rest_framework import status

def BAD_RESPONSE(message):
    return Response({"error": message}, status=status.HTTP_400_BAD_REQUEST)

def DATA_NOT_FOUND(message):
    return Response({"error": message}, status=status.HTTP_404_NOT_FOUND)

def Success_RESPONSE(query, result_dict):
    return Response({
        "query": query,
        "result": result_dict,
        "status": status.HTTP_200_OK
    })



def ProductResponse(message , matched_data , compare_df):
    return Response({
        "message": message ,
        "status": status.HTTP_200_OK if message =='success' else 404,
        "matched_data": [matched_data] if isinstance(matched_data,dict) else [],
        "compare_data": compare_df
    })