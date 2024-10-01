from flask import Flask, request, jsonify

from flasgger import Swagger
from query import query_rag

app = Flask(__name__)
swagger = Swagger(app)


@app.route('/query', methods=['GET'])
def query():
    """
    Query the system with a given text
    ---
    parameters:
      - name: query_text
        in: query
        type: string
        required: true
        description: The text to query the system.
    responses:
      200:
        description: The response from the RAG system
        schema:
          type: object
          properties:
            response:
              type: string
              description: The response text from the query
            message:
              type: string
              description: Error message, if any
      400:
        description: Bad request when query_text is missing
        schema:
          type: object
          properties:
            message:
              type: string
              description: Error message
      500:
        description: Internal server error
        schema:
          type: object
          properties:
            message:
              type: string
              description: Error message
    """
    query_text = request.args.get('query_text', '')
    if not query_text:
        return jsonify({'message': 'query_text parameter is required'}), 400

    try:
        response_text, sources = query_rag(query_text)
        return jsonify({'response': response_text,
                        'sources': sources})
    except Exception as e:
        return jsonify({'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
