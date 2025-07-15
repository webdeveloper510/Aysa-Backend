import os
from rest_framework import status
from rest_framework.response import Response



def DATA_NOT_FOUND(message : str):
    return Response({
        "message": message,
        'status': status.HTTP_404_NOT_FOUND
    })


def BAD_RESPONSE(message : str):
    return Response({
        "message": message,
        'status': status.HTTP_400_BAD_REQUEST
    })

def Success_RESPONSE(question : str , answer : str):
    return Response({
        "question": question ,
        "answer": answer,
        'status': status.HTTP_200_OK
    })