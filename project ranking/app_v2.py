from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

leaderboard = []

@app.route('/')
def index():
    # Ordenação decrescente pelo score global combinado
    ranking_ordenado = sorted(leaderboard, key=lambda x: x['score'], reverse=True)
    return render_template('index.html', ranking=ranking_ordenado)

@app.route('/submeter', methods=['POST'])
def submeter_resultado():
    dados = request.get_json()
    
    if not dados or 'grupo' not in dados or 'score' not in dados:
        return jsonify({'erro': 'Parâmetros inválidos'}), 400
    
    for item in leaderboard:
        if item['grupo'] == dados['grupo']:
            item['score'] = dados['score']
            item['score_reg'] = dados.get('score_reg', 0)
            item['score_class'] = dados.get('score_class', 0)
            return jsonify({'status': 'updated'}), 200
            
    leaderboard.append({
        'grupo': dados['grupo'],
        'score': dados['score'],
        'score_reg': dados.get('score_reg', 0),
        'score_class': dados.get('score_class', 0)
    })
    return jsonify({'status': 'created'}), 201

if __name__ == '__main__':
    app.run(debug=True, port=5000)